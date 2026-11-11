#!/usr/bin/env python3
"""Frontier/Dijkstra supervised DDQN; inference uses only the map-conditioned Student."""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppo_agent import HighLevelAgent
from sim2d_env import SCENARIOS, Sim2DEnv, export_world, make_layout
from util import (OnlineOccupancyMap, FrontierTeacher, SubgoalSpace, SubgoalProgress, high_state, HIGH_SPEC)

OUTCOMES = ('success','collision','timeout','no_safe_action')
TEACHER_REASONS = ('direct_goal','known_goal_route','frontier_exploration','abstain')
TRACE_FIELDS = ('scenario','seed','segment','student_action','student_greedy','executed_action','epsilon',
                'teacher_queried','teacher_action','teacher_valid','teacher_reason','teacher_detail','lambda_value',
                'exact_goal_available','exact_goal_selected','x_start','y_start','yaw_start','x_end','y_end','yaw_end',
                'subgoal_x','subgoal_y','steps','duration','reason','path_length','astar_before','astar_after',
                'r_arrival','r_collision','r_step','r_dist','r_no_safe_action','reward')


def candidate_state(env, space, mapping):
    points,mask,exact = space.candidates(env.state)
    c,s = math.cos(env.yaw),math.sin(env.yaw)
    world = points @ np.array([[c,s],[-s,c]]) + env.position
    state = high_state(mapping,env.position,env.yaw,env.goal_xy,env.velocity,exact)
    return state,mask,exact,world


def reset_episode(env, space, mapping, kind, seed):
    env.reset(kind,seed)
    if not OnlineOccupancyMap.inside(np.array([env.position,env.goal_xy])).all():
        raise ValueError('Task exceeds fixed map [-6,6)x[-6,6); select a compatible Gazebo route')
    mapping.reset()
    mapping.observe(env.ranges,env.position,env.yaw)
    _,mask,_,_ = candidate_state(env,space,mapping)
    if not mask.any():
        raise RuntimeError('Task initialization has no safe action: {} seed={}'.format(kind,seed))
    env.goal_seen = env.goal_near = False
    if env.show_memory:
        env.info.update(observed_map=mapping,candidate_points=np.empty((0,2)),
                        teacher_frontiers=np.empty((0,2)),teacher_path=np.empty((0,2)),
                        teacher_point=np.empty((0,2)),student_point=np.empty((0,2)))
    env.refresh()


def run_segment(agent, env, space, mapping, decisions, budget, deterministic=False, teacher=None, teacher_only=False):
    state,mask,exact,world = candidate_state(env,space,mapping)
    coefficient = agent.knowledge_coefficient(decisions) if not deterministic else 0.
    selection = dict(student_action=-1,student_greedy=-1,epsilon=0.,teacher_queried=False,
                     teacher_action=-1,teacher_valid=False,teacher_reason='not_used',teacher_detail='',lambda_value=coefficient)
    view = dict(frontiers=np.empty((0,2)),path=np.empty((0,2)),reference=np.empty((0,2)))
    if not teacher_only:
        index,student = agent.select_action(state,mask,decisions,deterministic)
        selection.update(student)
    if teacher_only or coefficient > 0.:
        label,valid,reason,view = teacher.recommend(state,world,mask)
        selection.update(teacher_queried=True,teacher_action=label,teacher_valid=valid,
                         teacher_reason=reason,teacher_detail=view['detail'])
    if teacher_only:
        if not selection['teacher_valid']:
            env.info.update(outcome='teacher_abstain')
            env.stop()
            return None,dict(selection,executed_action=-1,steps=0,duration=0.,reason='teacher_abstain',reward=0.)
        index = selection['teacher_action']
    selection.update(executed_action=index,exact_goal_available=exact>=0 and bool(mask[exact]),exact_goal_selected=index==exact)
    env.goal_seen |= exact>=0 and bool(mask[exact])
    env.goal_near |= env.info['distance']<.5
    env.set_subgoal(world[index])
    low_state = env.low_state()
    progress = SubgoalProgress(float(low_state[362])*5*math.sqrt(2),float(low_state[363])*math.pi,
                               .1 if index==exact else .3,env.v_max,env.w_max)
    before,path_before = env.astar_distance(),env.path_length
    start,yaw_start = env.position.copy(),float(env.yaw)
    elapsed = 0.
    for steps in range(1,budget+1):
        next_low,terminated,truncated,info = env.step(agent.low_action(low_state))
        mapping.observe(env.ranges,env.position,env.yaw)
        low_state = env.low_state()
        elapsed += info['sim_duration']
        reason = progress.update(float(low_state[362])*5*math.sqrt(2),float(low_state[363])*math.pi,elapsed)
        _,current_mask,current_exact = space.candidates(next_low)
        if not terminated and not current_mask.any():
            terminated,truncated = True,False
            info.update(outcome='no_safe_action',reached=False,collision=False,timeout=False)
            env.info.update(info)
            env.stop()
        visible = current_exact>=0 and bool(current_mask[current_exact])
        near = info['distance']<.5
        if index!=exact:
            if visible and not env.goal_seen:
                reason='goal_visible'
            elif near and not env.goal_near:
                reason='goal_near'
        env.goal_seen |= visible
        env.goal_near |= near
        if terminated or truncated:
            reason=info['outcome']
        elif steps==budget:
            reason='checkpoint_cut'
        env.info.update(subgoal_reason=reason,segment_steps=steps,segment_elapsed=elapsed,
                        subgoal_distance=float(low_state[362])*5*math.sqrt(2),
                        no_progress_time=elapsed-progress.last_progress,segment_time_limit=progress.limit,**selection)
        if env.show_memory:
            label=selection['teacher_action']
            env.info.update(observed_map=mapping,candidate_points=world[mask],teacher_frontiers=view['frontiers'],
                            teacher_path=view['path'],teacher_point=world[label:label+1] if label>=0 else np.empty((0,2)),
                            student_point=world[index:index+1])
        env.refresh()
        if terminated or truncated or reason!='running':
            break
    after=env.astar_distance()
    parts=dict(r_arrival=100. if info['reached'] else 0.,r_collision=-100. if info['collision'] else 0.,
               r_step=-.1*steps,r_dist=5*(before-after),r_no_safe_action=-100. if info['outcome']=='no_safe_action' else 0.)
    reward=float(sum(parts.values()))
    env.info.update(parts,segment_reward=reward)
    env.refresh()
    next_state,next_mask,_,_=candidate_state(env,space,mapping)
    if terminated:
        next_mask=np.zeros(225,bool)
    transition=dict(state=state,action=index,reward=reward,next_state=next_state,mask=mask,next_mask=next_mask,
                    terminated=terminated,truncated=truncated or (steps==budget and not terminated),steps=steps,
                    teacher_action=selection['teacher_action'],teacher_valid=selection['teacher_valid'])
    record=dict(selection,x_start=float(start[0]),y_start=float(start[1]),yaw_start=yaw_start,
                x_end=float(env.position[0]),y_end=float(env.position[1]),yaw_end=float(env.yaw),
                subgoal_x=float(env.subgoal_xy[0]),subgoal_y=float(env.subgoal_xy[1]),steps=steps,duration=elapsed,
                reason=reason,path_length=env.path_length-path_before,
                astar_before=before if isinstance(env,Sim2DEnv) else None,astar_after=after if isinstance(env,Sim2DEnv) else None,
                reward=reward,**parts)
    return transition,record


def evaluate(agent, env, space, config, suite='validation', decisions=0, trace_writer=None, teacher=None):
    teacher_only=config['eval_policy']=='teacher'
    count=100 if suite=='test' else config['eval_episodes']
    seed_base={'validation':5000000,'test':7000000}[suite]
    mapping=OnlineOccupancyMap(env.scan_angle_min,env.scan_angle_increment)
    rows=[]
    for i in range(count):
        kind=SCENARIOS[i%5] if config['scenario']=='all' else config['scenario']
        seed=seed_base+i
        reset_episode(env,space,mapping,kind,seed)
        reward,records=0.,[]
        while True:
            transition,record=run_segment(agent,env,space,mapping,decisions,env.timeout_steps-env.steps,
                                          deterministic=True,teacher=teacher,teacher_only=teacher_only)
            if trace_writer is not None:
                trace_writer.writerow(dict(record,scenario=kind,seed=seed,segment=len(records)+1))
            if transition is None:
                break
            reward+=transition['reward'];records.append(record)
            if transition['terminated'] or transition['truncated']:
                break
        info=env.info;reference=env.initial_distance
        spl=min(1.,reference/max(reference,env.path_length)) if info['reached'] and reference>0 else 0.
        rows.append(dict(scenario=kind,seed=seed,outcome=info['outcome'],steps=env.steps,distance=info['distance'],
                         path_length=env.path_length,reference_distance=reference,spl=spl,reward=reward,subgoals=len(records),
                         stalled=sum(r['reason']=='stalled' for r in records),
                         remaining_astar=env.astar_distance() if isinstance(env,Sim2DEnv) else None))
    outcomes=OUTCOMES+('teacher_abstain',) if teacher_only else OUTCOMES
    rates={name:sum(row['outcome']==name for row in rows)/count for name in outcomes}
    message='teacher_eval' if teacher_only else 'eval'
    message+=' episodes={} success={:.1%} collision={:.1%} timeout={:.1%} no_safe_action={:.1%}'
    message=message.format(count,*[rates[name] for name in OUTCOMES])
    if teacher_only:
        message+=' teacher_abstain={:.1%}'.format(rates['teacher_abstain'])
    print(message,flush=True)
    return rates,rows


def write_rows(path, rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def write_metrics(directory, records):
    records.sort(key=lambda r:(r['eval_success_rate'] if r['eval_episodes'] else -1.,
                               r['eval_mean_spl'] if r['eval_episodes'] else -1.,r['total_steps']),reverse=True)
    write_rows(directory/'checkpoint_metrics.csv',records)


def save_evaluate(agent, env, space, config, counters, outcomes, run_dir, records, writer, final):
    counters['recent_outcomes']=list(outcomes)
    path=run_dir/'step_{:06d}.pt'.format(counters['total_steps'])
    agent.save(path,counters)
    if final:
        agent.save(run_dir/'last.pt',counters)
    row=dict(checkpoint=path.name,total_steps=counters['total_steps'],episode=counters['episode'],
             high_decisions=counters['high_decisions'],gradient_updates=agent.updates,target_syncs=agent.target_syncs,
             window_episodes=len(outcomes),knowledge_lambda=agent.knowledge_coefficient(counters['high_decisions']))
    row.update({name+'_rate_100':outcomes.count(name)/len(outcomes) if outcomes else None for name in OUTCOMES})
    row.update(eval_episodes=0,eval_success_rate=None,eval_collision_rate=None,eval_timeout_rate=None,
               eval_no_safe_action_rate=None,eval_mean_spl=None,eval_mean_steps=None,eval_mean_path_length=None,
               eval_mean_subgoals=None,**{kind+'_success_rate':None for kind in SCENARIOS})
    records.append(row);write_metrics(run_dir,records)
    start=time.perf_counter()
    rates,rows=evaluate(agent,env,space,config,decisions=counters['high_decisions'])
    counters['eval_wall_seconds']+=time.perf_counter()-start
    counters['eval_steps']+=sum(row['steps'] for row in rows)
    row.update(eval_episodes=len(rows),eval_mean_spl=float(np.mean([r['spl'] for r in rows])),
               eval_mean_steps=float(np.mean([r['steps'] for r in rows])),
               eval_mean_path_length=float(np.mean([r['path_length'] for r in rows])),
               eval_mean_subgoals=float(np.mean([r['subgoals'] for r in rows])),
               **{'eval_'+name+'_rate':rate for name,rate in rates.items()})
    for kind in SCENARIOS:
        subset=[r for r in rows if r['scenario']==kind]
        row[kind+'_success_rate']=sum(r['outcome']=='success' for r in subset)/len(subset) if subset else None
    write_rows(run_dir/'eval_tasks_step_{:06d}.csv'.format(counters['total_steps']),rows)
    write_metrics(run_dir,records)
    for name,rate in rates.items():writer.add_scalar('eval/'+name+'_rate',rate,counters['total_steps'])
    for name in ('spl','steps','path_length','subgoals'):
        writer.add_scalar('eval/'+('spl' if name=='spl' else 'mean_'+name),float(np.mean([r[name] for r in rows])),counters['total_steps'])
    (run_dir/'progress.json').write_text(json.dumps(counters,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('train','eval'),default='train')
    parser.add_argument('--backend',choices=('sim2d','gazebo'),default='sim2d')
    parser.add_argument('--guidance',choices=('on','off'),default='on')
    parser.add_argument('--eval-policy',choices=('student','teacher'),default='student')
    parser.add_argument('--eval-suite',choices=('validation','test'),default='validation')
    parser.add_argument('--render',choices=('on','off'),default='off')
    parser.add_argument('--trace',choices=('on','off'),default='off')
    parser.add_argument('--scenario',choices=('all',)+SCENARIOS,default='all')
    parser.add_argument('--scenario-seed',type=int,default=3)
    parser.add_argument('--seed',type=int,default=3)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--low-checkpoint',type=Path,default=Path(__file__).resolve().parent/'runs/attention_paper/step_180000.pt')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'runs')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--max-train-steps',type=int,default=200000)
    parser.add_argument('--save-interval',type=int,default=10000)
    parser.add_argument('--eval-episodes',type=int,default=20)
    parser.add_argument('--timeout-steps',type=int,default=800)
    parser.add_argument('--high-lr',type=float,default=3e-4)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--replay-capacity',type=int,default=50000)
    parser.add_argument('--warmup',type=int,default=256)
    parser.add_argument('--updates-per-decision',type=int,default=4)
    parser.add_argument('--target-sync-decisions',type=int,default=200)
    parser.add_argument('--epsilon-initial',type=float,default=.3)
    parser.add_argument('--epsilon-final',type=float,default=.02)
    parser.add_argument('--epsilon-decay-decisions',type=int,default=2000)
    parser.add_argument('--gamma',type=float,default=.99)
    parser.add_argument('--discount-reference-steps',type=float,default=25.)
    parser.add_argument('--max-grad-norm',type=float,default=.5)
    parser.add_argument('--lambda-initial',type=float,default=.5)
    parser.add_argument('--lambda-decay-decisions',type=int,default=2000)
    parser.add_argument('--teacher-temperature',type=float,default=1.)
    parser.add_argument('--teacher-q-normalization',choices=('on','off'),default='on')
    parser.add_argument('--frontier-weights',type=float,nargs=3,default=(1.,1.,1.))
    parser.add_argument('--teacher-deviation-weight',type=float,default=1.)
    parser.add_argument('--teacher-min-progress',type=float,default=.05)
    parser.add_argument('--start',type=float,nargs=2)
    parser.add_argument('--goal',type=float,nargs=2)
    parser.add_argument('--yaw',type=float,default=180.)
    parser.add_argument('--robot-model-name',default='turtlebot3')
    parser.add_argument('--sensor-timeout',type=float,default=10.)
    parser.add_argument('--export-world',type=Path)
    args=parser.parse_args()
    for key in ('max_train_steps','save_interval','eval_episodes','timeout_steps','high_lr','batch_size','replay_capacity',
                'warmup','updates_per_decision','target_sync_decisions','epsilon_decay_decisions','discount_reference_steps',
                'max_grad_norm','lambda_decay_decisions','teacher_temperature','sensor_timeout'):
        value=getattr(args,key)
        if not np.isfinite(value) or value<=0:parser.error('--'+key.replace('_','-')+' must be finite and positive')
    if not args.batch_size<=args.warmup<=args.replay_capacity or not 0<=args.gamma<=1:
        parser.error('Require batch-size <= warmup <= replay-capacity and gamma in [0,1]')
    if not np.isfinite([args.epsilon_initial,args.epsilon_final]).all() or not 0<=args.epsilon_final<=args.epsilon_initial<=1:
        parser.error('Require 0 <= epsilon-final <= epsilon-initial <= 1')
    for key in ('lambda_initial','teacher_deviation_weight','teacher_min_progress'):
        value=getattr(args,key)
        if not np.isfinite(value) or value<0:parser.error('--'+key.replace('_','-')+' must be finite and nonnegative')
    if not np.isfinite(args.frontier_weights).all() or min(args.frontier_weights)<0 or sum(args.frontier_weights)==0:
        parser.error('Frontier weights must be finite, nonnegative and not all zero')
    if args.seed<0 or args.scenario_seed<0:parser.error('Seeds must be nonnegative')
    if (args.start is None)!=(args.goal is None):parser.error('--start and --goal must be supplied together')
    if not np.isfinite([args.yaw]+(args.start or [])+(args.goal or [])).all():parser.error('Route values must be finite')
    if args.start is not None:
        if args.start==args.goal:parser.error('Start and goal must differ')
        if not OnlineOccupancyMap.inside(np.array([args.start,args.goal])).all():parser.error('Route outside fixed map [-6,6)x[-6,6)')
    if args.mode=='train' and args.eval_policy!='student':parser.error('Teacher-only is an explicit evaluation diagnostic')
    if args.mode=='train' and args.eval_suite!='validation':parser.error('Training uses the validation suite')
    if args.export_world:
        if args.scenario=='all':parser.error('--export-world requires one scenario')
        start,goal,yaw=export_world(args.export_world,args.scenario,args.scenario_seed)
        print('world={} start={} goal={} yaw={}'.format(args.export_world.resolve(),start.tolist(),goal.tolist(),math.degrees(yaw)))
        return
    if args.backend=='sim2d' and args.start is not None:parser.error('Explicit route is for Gazebo only')
    if args.backend=='gazebo' and (args.mode!='eval' or (args.scenario=='all' and args.start is None)):
        parser.error('Gazebo requires evaluation and a single scenario or explicit route')
    if args.backend=='gazebo' and args.eval_suite!='validation':parser.error('The fixed test suite requires Sim2D')
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable; use --device cpu')
    path=args.checkpoint or args.low_checkpoint
    if not path.is_file():parser.error('Checkpoint not found: '+str(path))
    if args.mode=='eval' and args.eval_policy=='student' and args.checkpoint is None:
        parser.error('Student evaluation requires --checkpoint')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if args.checkpoint:
        if (saved.get('algorithm')!='FRONTIER_DDQN' or saved['version']!=1
                or saved['input_spec']!=HIGH_SPEC or saved['map_config']!=OnlineOccupancyMap.config):
            parser.error('Requires FRONTIER_DDQN v1; use legacy_high for old models')
        for key in ('high_lr','batch_size','replay_capacity','warmup','updates_per_decision','target_sync_decisions',
                    'epsilon_initial','epsilon_final','epsilon_decay_decisions','gamma','discount_reference_steps','max_grad_norm',
                    'lambda_initial','lambda_decay_decisions','teacher_temperature','teacher_q_normalization',
                    'frontier_weights','teacher_deviation_weight','teacher_min_progress','guidance'):
            setattr(args,key,saved['config'][key])
        if args.mode=='train':
            for key in ('seed','scenario','timeout_steps'):
                if getattr(args,key)!=saved['config'][key]:parser.error('Resume requires unchanged --'+key.replace('_','-'))
        low_checkpoint,low_source=saved['low_checkpoint'],saved['low_source']
    else:
        low_checkpoint=saved
        low_source=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(1)
    agent=HighLevelAgent(config,low_checkpoint,low_source)
    counters=dict(total_steps=0,high_decisions=0,episode=0,tasks=0,recent_outcomes=[],
                  teacher_queries=0,teacher_valid=0,teacher_agreements=0,
                  teacher_reasons={key:0 for key in TEACHER_REASONS},
                  train_wall_seconds=0.,eval_wall_seconds=0.,eval_steps=0)
    if args.checkpoint:
        if args.mode=='train':agent.restore(saved)
        else:agent.online.load_state_dict(saved['online'])
        counters=saved['counters']
    if args.mode=='train' and counters['total_steps']>=args.max_train_steps:parser.error('Budget must exceed restored total steps')
    # Teacher lifetime is selected once; normal evaluation never instantiates it.
    needs_teacher=(args.mode=='train' and args.guidance=='on' and args.lambda_initial>0) or args.eval_policy=='teacher'
    teacher=FrontierTeacher(args.frontier_weights,args.teacher_deviation_weight,args.teacher_min_progress) if needs_teacher else None
    low_config=low_checkpoint['config'];space=SubgoalSpace(low_config['scan_angle_min'],low_config['scan_angle_increment'])
    with ExitStack() as resources:
        if args.backend=='sim2d':
            env=Sim2DEnv(low_config,args.timeout_steps,args.render=='on')
        else:
            import rospy
            from turtlebot3_env import HighGazeboEnv
            rospy.init_node('frontier_ddqn',anonymous=True)
            if args.start is None:_,start,goal,yaw=make_layout(args.scenario,args.scenario_seed)
            else:start,goal,yaw=args.start,args.goal,math.radians(args.yaw)
            env=HighGazeboEnv(low_config,start,goal,yaw,args.timeout_steps,args.robot_model_name,args.sensor_timeout)
        env.show_memory=args.backend=='sim2d' and args.render=='on'
        resources.callback(env.close)
        if args.mode=='eval':
            args.output_dir.mkdir(parents=True,exist_ok=True)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            prefix='teacher_diagnostic' if args.eval_policy=='teacher' else 'student_evaluation'
            trace_writer=None
            if args.trace=='on':
                handle=resources.enter_context((args.output_dir/(prefix+'_segments_'+stamp+'.csv')).open('w',newline=''))
                trace_writer=csv.DictWriter(handle,fieldnames=TRACE_FIELDS);trace_writer.writeheader()
            _,rows=evaluate(agent,env,space,config,args.eval_suite,counters['high_decisions'],trace_writer,teacher)
            write_rows(args.output_dir/(prefix+'_'+stamp+'.csv'),rows)
            return
        from torch.utils.tensorboard import SummaryWriter
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_dir=args.output_dir/('frontier_ddqn_'+stamp);run_dir.mkdir(parents=True)
        writer=resources.enter_context(SummaryWriter(str(run_dir)))
        trace_writer=None
        if args.trace=='on':
            handle=resources.enter_context((run_dir/'segments.csv').open('w',newline=''))
            trace_writer=csv.DictWriter(handle,fieldnames=TRACE_FIELDS);trace_writer.writeheader()
        (run_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        outcomes=deque(counters['recent_outcomes'],maxlen=100);records=[]
        mapping=OnlineOccupancyMap(env.scan_angle_min,env.scan_angle_increment)
        run_started=time.perf_counter()
        kind=SCENARIOS[counters['tasks']%5] if args.scenario=='all' else args.scenario
        reset_episode(env,space,mapping,kind,args.seed+counters['tasks']);counters['tasks']+=1
        episode_reward=0.;train_started=time.perf_counter()
        while counters['total_steps']<args.max_train_steps:
            total=counters['total_steps']
            budget=min(args.save_interval-total%args.save_interval,args.max_train_steps-total,env.timeout_steps-env.steps)
            transition,record=run_segment(agent,env,space,mapping,counters['high_decisions'],budget,teacher=teacher)
            counters['total_steps']+=transition['steps'];counters['high_decisions']+=1
            episode_reward+=transition['reward'];agent.replay.add(**transition)
            if record['teacher_queried']:
                counters['teacher_queries']+=1;counters['teacher_valid']+=int(record['teacher_valid'])
                counters['teacher_reasons'][record['teacher_reason']]+=1
                counters['teacher_agreements']+=int(record['teacher_valid'] and record['teacher_action']==record['student_greedy'])
            if trace_writer is not None:trace_writer.writerow(dict(record,scenario=env.kind,seed=env.seed,segment=counters['high_decisions']))
            writer.add_scalar('loss/lambda',agent.knowledge_coefficient(counters['high_decisions']),counters['total_steps'])
            if counters['teacher_queries']:
                stats=dict(valid_rate=counters['teacher_valid']/counters['teacher_queries'],
                           agreement_with_student=counters['teacher_agreements']/max(counters['teacher_valid'],1),
                           active=float(agent.knowledge_coefficient(counters['high_decisions'])>0))
                stats.update({dict(direct_goal='direct_goal_rate',known_goal_route='known_goal_route_rate',frontier_exploration='frontier_rate',abstain='abstain_rate')[key]:count/counters['teacher_queries'] for key,count in counters['teacher_reasons'].items()})
                for name,value in stats.items():writer.add_scalar('teacher/'+name,value,counters['total_steps'])
            if agent.replay.size>=args.warmup:
                losses=agent.learn(counters['high_decisions'])
                for name,value in losses.items():writer.add_scalar('loss/'+name,value,counters['total_steps'])
            for name,value in dict(epsilon=record['epsilon'],decisions=counters['high_decisions'],gradient_updates=agent.updates,target_syncs=agent.target_syncs).items():
                writer.add_scalar('high/'+name,value,counters['total_steps'])
            complete=env.info['outcome']!='running'
            if complete:
                counters['episode']+=1;outcomes.append(env.info['outcome'])
                writer.add_scalar('train/episode_reward',episode_reward,counters['episode'])
                for name in OUTCOMES:writer.add_scalar('train/'+name+'_rate',outcomes.count(name)/len(outcomes),counters['episode'])
            final=counters['total_steps']==args.max_train_steps
            due=counters['total_steps']%args.save_interval==0 or final
            if due:
                counters['train_wall_seconds']+=time.perf_counter()-train_started
                save_evaluate(agent,env,space,config,counters,outcomes,run_dir,records,writer,final)
                train_started=time.perf_counter()
            if (complete or due) and not final:
                kind=SCENARIOS[counters['tasks']%5] if args.scenario=='all' else args.scenario
                reset_episode(env,space,mapping,kind,args.seed+counters['tasks']);counters['tasks']+=1;episode_reward=0.
        timing=dict(train_wall_seconds=counters['train_wall_seconds'],eval_wall_seconds=counters['eval_wall_seconds'],
                    total_wall_seconds=time.perf_counter()-run_started,total_steps=counters['total_steps'],
                    eval_steps=counters['eval_steps'],high_decisions=counters['high_decisions'],gradient_updates=agent.updates)
        (run_dir/'timing.json').write_text(json.dumps(timing,indent=2)+'\n')


if __name__=='__main__':
    main()
