"""Small numerical/regression checks; no training-performance claims."""
import ast
import contextlib
import copy
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import main
from ppo_agent import HighLevelAgent
from sim2d_env import SCENARIOS, Sim2DEnv, raycast
from util import (OnlineOccupancyMap as Map, FrontierTeacher, SimpleDuelingQNetwork,
                  HighReplayBuffer, high_state, state_tensors, teacher_cross_entropy, SubgoalSpace)


def config(**overrides):
    result=dict(device='cpu',high_lr=3e-4,replay_capacity=32,batch_size=2,warmup=2,
                updates_per_decision=2,target_sync_decisions=2,epsilon_initial=.3,epsilon_final=.02,
                epsilon_decay_decisions=20,gamma=.99,discount_reference_steps=25.,max_grad_norm=.5,
                lambda_initial=.5,lambda_decay_decisions=4,guidance='on',teacher_temperature=1.,
                teacher_q_normalization='on',eval_policy='student',eval_episodes=1,scenario='all')
    result.update(overrides)
    return result


def state(occupancy=None,position=(.1,.1),goal=(4.1,.1),exact=-1):
    mapping=Map(0.,2*math.pi/360)
    if occupancy is not None:
        mapping.seen[:]=occupancy!=-1
        mapping.log_odds[:]=np.where(occupancy==0,-1.,1.)
    return high_state(mapping,np.array(position),0.,np.array(goal),np.zeros(2),exact)


def add(replay,s,**kwargs):
    mask=np.ones(225,bool)
    values=dict(state=s,action=0,reward=1.,next_state=s,mask=mask,next_mask=mask,
                terminated=False,truncated=False,steps=25,teacher_action=1,teacher_valid=True)
    values.update(kwargs)
    replay.add(**values)


class MapTests(unittest.TestCase):
    def test_coordinates_and_bounds(self):
        np.testing.assert_allclose(Map.centers([[0,0],[59,59]]),[[-5.9,-5.9],[5.9,5.9]])
        np.testing.assert_array_equal(Map.cells([[-6.,-6.],[5.999,5.999],[.1,.3]]),[[0,0],[59,59],[31,30]])
        np.testing.assert_array_equal(Map.inside([[-6,-6],[6,0],[0,-6.001]]),[True,False,False])

    def test_seen_evidence_and_reset(self):
        m=Map(0.,2*math.pi/360)
        self.assertTrue((m.snapshot()==-1).all())
        m.seen[2,3]=True
        self.assertEqual(m.snapshot()[2,3],1)
        m.log_odds[2,3]=-.1
        self.assertEqual(m.snapshot()[2,3],0)
        m.reset()
        self.assertTrue((m.snapshot()==-1).all())

    def test_one_update_per_frame_hit_priority_range(self):
        m=Map(0.,0.)
        m.observe(np.ones(360),np.array([0.,0.]),0.)
        self.assertAlmostEqual(float(m.log_odds[30,30]),-.4,places=6)
        self.assertAlmostEqual(float(m.log_odds[tuple(Map.cells([.968,0.]))]),.85,places=6)
        m.reset();m.observe(np.full(360,3.5),np.array([0.,0.]),0.)
        self.assertFalse((m.snapshot()==1).any())
        m.seen[30,30]=False
        before=m.log_odds.copy()
        with self.assertRaisesRegex(ValueError,'no observations were cropped'):
            m.observe(np.full(360,3.5),np.array([5.,0.]),0.)
        np.testing.assert_array_equal(m.log_odds,before)

    def test_state_normalization_and_exact_slot(self):
        s=state(position=(-3.,1.),goal=(3.,-2.),exact=224)
        self.assertEqual(s['map'].size+s['robot'].size,3610)
        _,r=state_tensors(s,'cpu',(.2,1.5))
        np.testing.assert_allclose(r[0,:2],[-.5,1/6])
        np.testing.assert_array_equal(r[0,-2:],[1.,1.])


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.teacher=FrontierTeacher()

    def test_dijkstra_cost_unknown_and_corner(self):
        free=np.ones((5,5),bool)
        d,_=self.teacher.search(free,(0,0))
        self.assertAlmostEqual(d[3,2],.2*(2*math.sqrt(2)+1))
        free[:]=False;free[0,0]=free[1,1]=True
        d,_=self.teacher.search(free,(0,0))
        self.assertTrue(np.isinf(d[1,1]))
        free[0,0]=False
        self.assertTrue(np.isinf(self.teacher.search(free,(0,0))[0]).all())

    def test_frontier_four_neighbors_information_at_boundary(self):
        occ=np.ones((60,60),np.int8);occ[10:13,10:13]=0;occ[9,9]=-1
        free=self.teacher.planning_free(occ)
        self.assertEqual(len(self.teacher.frontiers(occ,free,free)),0)
        occ[9,10]=-1
        cells=self.teacher.frontiers(occ,free,free)
        np.testing.assert_array_equal(cells,[[10,10]])
        self.assertEqual(self.teacher.information_gain(np.full((60,60),-1,np.int8),np.array([[0,0]]))[0],1.)

    def test_center_distance_inflation_not_rounded_up(self):
        occ=np.zeros((60,60),np.int8);occ[30,30]=1
        free=self.teacher.planning_free(occ)
        self.assertFalse(free[30,30]);self.assertTrue(free[30,31])
        self.assertEqual((~free).sum(),1)

    def test_exact_goal_and_abstention(self):
        s=state(exact=100)
        world=np.zeros((225,2));mask=np.zeros(225,bool);mask[100]=True
        label,valid,reason,_=self.teacher.recommend(s,world,mask)
        self.assertEqual((label,valid,reason),(100,True,'direct_goal'))
        s['robot'][-1]=0
        label,valid,_,view=self.teacher.recommend(s,world,mask)
        self.assertFalse(valid);self.assertEqual(label,-1);self.assertEqual(view['detail'],'robot_not_free')

    def test_known_goal_and_positive_progress(self):
        s=state(np.zeros((60,60),np.int8))
        world=np.tile([-.9,.1],(225,1));world[7]=[1.1,.1];world[8]=[1.1,.1]
        mask=np.zeros(225,bool);mask[[2,7,8]]=True
        action,valid,reason,view=self.teacher.recommend(s,world,mask)
        self.assertTrue(valid);self.assertEqual(action,7);self.assertEqual(reason,'known_goal_route')
        self.assertGreater(len(view['path']),1)
        self.assertGreater(np.linalg.norm(view['reference'][0]-s['robot'][:2]),3.5)
        mask[:]=False
        self.assertFalse(self.teacher.recommend(s,world,mask)[1])

    def test_far_frontier_and_dead_end_backtracking(self):
        occ=np.ones((60,60),np.int8)
        occ[20:46,29:32]=0;occ[46,29:32]=-1
        s=state(occ,position=(.1,-1.9),goal=(.1,-4.1))
        # Opening lies behind the robot and away from its ultimate goal.
        world=np.tile([.1,-2.5],(225,1));world[5]=[.1,-.9]
        mask=np.zeros(225,bool);mask[[0,5]]=True
        action,valid,reason,view=self.teacher.recommend(s,world,mask)
        self.assertEqual((action,valid,reason),(5,True,'frontier_exploration'))
        self.assertGreater(np.linalg.norm(view['reference'][0]-s['robot'][:2]),3.5)
        self.assertGreater(np.linalg.norm(world[action]-s['robot'][4:6]),np.linalg.norm(s['robot'][:2]-s['robot'][4:6]))

    def test_path_segment_distance(self):
        d=self.teacher.path_deviation(np.array([[1.,1.],[3.,0.]]),np.array([[0.,0.],[2.,0.]]))
        np.testing.assert_allclose(d,[1.,1.])


class NetworkReplayTests(unittest.TestCase):
    def test_batch_1_64_and_all_advantage_mean(self):
        net=SimpleDuelingQNetwork()
        for count in (1,64):
            q=net(torch.zeros(count,1,60,60),torch.zeros(count,10))
            self.assertEqual(q.shape,(count,225));q.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))
            net.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p in net.parameters():p.zero_()
            net.advantage[-1].bias.copy_(torch.arange(225))
        q=net(torch.zeros(1,1,60,60),torch.zeros(1,10))
        self.assertAlmostEqual(float(q.mean()),0.)
        self.assertEqual(float(q[0,0]),-112.)

    def test_masked_cross_entropy_gradients_zero_labels(self):
        q=torch.tensor([[1.,2.,100.],[0.,0.,0.]],requires_grad=True)
        mask=torch.tensor([[True,True,False],[True,False,False]])
        loss=teacher_cross_entropy(q,mask,torch.tensor([1,-1]),torch.tensor([True,False]),1.,True)
        loss.backward()
        self.assertEqual(float(q.grad[0,2]),0.);self.assertTrue((q.grad[1]==0).all())
        self.assertEqual(float(teacher_cross_entropy(q,mask,torch.tensor([-1,-1]),torch.zeros(2,dtype=torch.bool),1.,True)),0.)
        self.assertTrue(torch.isfinite(loss))

    def test_replay_snapshots_encoding_memory_and_roundtrip(self):
        b=HighReplayBuffer(3);s=state()
        add(b,s);s['map'][:]=1;s['robot'][:]=4
        self.assertTrue((b.maps[0]==0).all());self.assertNotEqual(b.robots[0,0],4.)
        self.assertEqual(b.maps.dtype,np.uint8)
        per_row=(b.maps.nbytes+b.next_maps.nbytes+b.robots.nbytes+b.next_robots.nbytes+sum(a.nbytes for a in b.data.values()))/3
        self.assertLess(per_row*50000,400_000_000)
        for _ in range(4):add(b,s)
        restored=HighReplayBuffer(3);restored.load_state_dict(b.state_dict())
        self.assertEqual(restored.position,b.position)
        np.testing.assert_array_equal(restored.maps,b.maps)
        batch=restored.sample(2,'cpu',(.2,1.5))
        self.assertTrue((batch['states'][0]==1.).all())


class AgentFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.low=torch.load(ROOT/'runs/attention_paper/step_180000.pt',map_location='cpu',weights_only=False)

    def agent(self,**kwargs):
        return HighLevelAgent(config(**kwargs),self.low,dict(path='test',sha256='test'))

    def test_schedule_mask_uniform_independent_and_low_frozen(self):
        a=self.agent()
        self.assertEqual(a.knowledge_coefficient(2),.5);self.assertEqual(a.knowledge_coefficient(6),0.)
        self.assertEqual(self.agent(guidance='off').knowledge_coefficient(0),0.)
        s=state();mask=np.zeros(225,bool);mask[[3,4]]=True
        action,_=a.select_action(s,mask,0,True);self.assertIn(action,(3,4))
        a.config['epsilon_initial']=1.;a.config['epsilon_final']=1.
        actions={a.select_action(s,mask,0)[0] for _ in range(30)}
        self.assertEqual(actions,{3,4})
        before={name:copy.deepcopy(getattr(a.low,name).state_dict()) for name in ('actor','critic','fuzzy','fusion')}
        for _ in range(2):add(a.replay,s)
        losses=a.learn(2);a.learn(3)
        self.assertEqual(a.updates,4);self.assertEqual(a.target_syncs,1)
        self.assertTrue(all(np.isfinite(list(losses.values()))))
        for name,weights in before.items():
            for key,value in weights.items():self.assertTrue(torch.equal(value,getattr(a.low,name).state_dict()[key]))
            self.assertTrue(all(not p.requires_grad for p in getattr(a.low,name).parameters()))

    def test_double_dqn_mask_termination_and_duration_bootstrap(self):
        a=self.agent(batch_size=3)
        class FixedQ(nn.Module):
            def __init__(self,values):
                super().__init__();self.values=nn.Parameter(torch.tensor(values,dtype=torch.float32))
            def forward(self,m,r):return self.values.expand(len(m),-1)
        a.online=FixedQ([0.,3.,100.]);a.target=FixedQ([20.,7.,500.])
        a.optimizer=torch.optim.Adam(a.online.parameters(),lr=1e-3)
        m=torch.tensor([[True,True,False]]*3)
        batch=dict(states=(torch.zeros(3,1,60,60),torch.zeros(3,10)),next_states=(torch.zeros(3,1,60,60),torch.zeros(3,10)),
                   masks=m,next_masks=torch.tensor([[False]*3,[True,True,False],[False]*3]),
                   rewards=torch.ones(3),actions=torch.zeros(3,dtype=torch.long),steps=torch.tensor([25,50,25]),
                   terminated=torch.tensor([True,False,False]),truncated=torch.tensor([False,True,False]))
        with patch.object(a.replay,'sample',return_value=batch):loss=a.update(0.)
        expected=torch.nn.functional.smooth_l1_loss(torch.zeros(3),torch.tensor([1.,1.+.99**2*7.,1.]))
        self.assertAlmostEqual(loss['td'],float(expected),places=5)

    def test_checkpoint_complete_roundtrip(self):
        a=self.agent();s=state()
        add(a.replay,s);add(a.replay,s);a.learn(2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';a.save(path,dict(total_steps=12,high_decisions=2))
            saved=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual((saved['algorithm'],saved['version']),('FRONTIER_DDQN',1))
            b=self.agent();b.restore(saved)
            self.assertEqual(b.replay.size,2);self.assertEqual(b.updates,a.updates)
            for k,v in a.online.state_dict().items():self.assertTrue(torch.equal(v,b.online.state_dict()[k]))
            self.assertTrue(b.optimizer.state_dict()['state'])

    def test_five_scene_segments_and_student_evaluation_no_teacher(self):
        a=self.agent();space=SubgoalSpace(self.low['config']['scan_angle_min'],self.low['config']['scan_angle_increment'])
        env=Sim2DEnv(self.low['config'],timeout_steps=3);env.show_memory=False
        mapping=Map(env.scan_angle_min,env.scan_angle_increment);teacher=FrontierTeacher()
        for i,kind in enumerate(SCENARIOS):
            main.reset_episode(env,space,mapping,kind,3+i)
            transition,record=main.run_segment(a,env,space,mapping,0,3,teacher=teacher)
            self.assertEqual(transition['action'],record['student_action'])
            self.assertTrue(transition['mask'][transition['action']])
            if record['teacher_valid']:
                self.assertTrue(transition['mask'][record['teacher_action']])
            self.assertGreater(transition['steps'],0)
            a.replay.add(**transition)
        before=copy.deepcopy(a.online.state_dict())
        with patch.object(FrontierTeacher,'__init__',side_effect=AssertionError('Teacher instantiated')),patch.object(FrontierTeacher,'search',side_effect=AssertionError('Planner called')),contextlib.redirect_stdout(io.StringIO()):
            rates,rows=main.evaluate(a,env,space,config())
        self.assertEqual(sum(rates.values()),1.)
        for k,v in before.items():self.assertTrue(torch.equal(v,a.online.state_dict()[k]))
        self.assertLessEqual(rows[0]['steps'],3)

    def test_no_safe_action_priority_and_initial_error(self):
        a=self.agent();space=SubgoalSpace(self.low['config']['scan_angle_min'],self.low['config']['scan_angle_increment'])
        env=Sim2DEnv(self.low['config'],timeout_steps=1);env.show_memory=False
        mapping=Map(env.scan_angle_min,env.scan_angle_increment)
        main.reset_episode(env,space,mapping,'u_wall',3)
        original=space.candidates;calls=0
        def empty_after_start(s):
            nonlocal calls
            calls+=1
            p,m,e=original(s)
            return p,(m if calls==1 else np.zeros(225,bool)),e
        with patch.object(space,'candidates',side_effect=empty_after_start):
            transition,record=main.run_segment(a,env,space,mapping,10,1,deterministic=True)
        self.assertEqual(record['reason'],'no_safe_action')
        self.assertTrue(transition['terminated']);self.assertFalse(transition['truncated']);self.assertFalse(transition['next_mask'].any())
        self.assertEqual(record['r_no_safe_action'],-100.)
        self.assertAlmostEqual(record['reward'],sum(record[k] for k in ('r_arrival','r_collision','r_dist','r_step','r_no_safe_action')))
        with patch.object(space,'candidates',return_value=(space.points,np.zeros(225,bool),-1)):
            with self.assertRaisesRegex(RuntimeError,'initialization'):
                main.reset_episode(env,space,mapping,'u_wall',3)

    def test_teacher_diagnostic_abstains_without_student_fallback(self):
        a=self.agent();space=SubgoalSpace(self.low['config']['scan_angle_min'],self.low['config']['scan_angle_increment'])
        env=Sim2DEnv(self.low['config'],timeout_steps=1);env.show_memory=False
        teacher=FrontierTeacher()
        with patch.object(teacher,'recommend',return_value=(-1,False,'abstain',dict(detail='test'))),patch.object(a,'select_action',side_effect=AssertionError('Student fallback')),contextlib.redirect_stdout(io.StringIO()):
            rates,rows=main.evaluate(a,env,space,config(eval_policy='teacher'),teacher=teacher)
        self.assertEqual(rates['teacher_abstain'],1.);self.assertEqual(rows[0]['steps'],0)

    def test_cached_viewer_five_scenes_and_close(self):
        import matplotlib
        matplotlib.use('Agg')
        import warnings
        with patch.object(matplotlib,'use'),warnings.catch_warnings(record=True) as caught:
            env=Sim2DEnv(self.low['config'],timeout_steps=1,render=True);env.show_memory=True
            space=SubgoalSpace(env.scan_angle_min,env.scan_angle_increment)
            mapping=Map(env.scan_angle_min,env.scan_angle_increment)
            for i,kind in enumerate(SCENARIOS):
                main.reset_episode(env,space,mapping,kind,3+i)
                main.run_segment(self.agent(),env,space,mapping,0,1,teacher=FrontierTeacher())
            env.viewer.figure.savefig('/tmp/frontier_v5_viewer.png')
            env.viewer.close()
            env.step(np.zeros(2))
            env.refresh()
            env.close()
        self.assertFalse(any('Legend does not support' in str(w.message) for w in caught))

    def test_low_source_and_ros_unchanged(self):
        for file,classes in [('util.py',('Actor','Critic','FuzzyKnowledge','HyperFusion','compute_gae','goal_observation','SubgoalSpace','SubgoalProgress')),
                             ('ppo_agent.py',('PPOAgent',))]:
            old=ast.parse((ROOT/'legacy_high'/file).read_text());new=ast.parse((ROOT/file).read_text())
            old={n.name:ast.dump(n) for n in old.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
            new={n.name:ast.dump(n) for n in new.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
            for name in classes:self.assertEqual(old[name],new[name],name)
        for file in ('eval.py','turtlebot3_env.py'):
            self.assertEqual((ROOT/file).read_bytes(),(ROOT/'legacy_high'/file).read_bytes())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
