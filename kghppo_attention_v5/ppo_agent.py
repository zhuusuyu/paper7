"""PPO-Clip over one spatial-attention / fuzzy / hypernetwork Gaussian policy."""

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import functional as F

from util import Actor, Critic, FuzzyKnowledge, HyperFusion, compute_gae


class PPOAgent:
    def __init__(self, config, state_dim=364):
        self.config = dict(config)
        self.device = torch.device(config["device"])
        self.actor = Actor(config["net_width"], config["v_max"], config["w_max"],
                           config["scan_angle_min"], config["scan_angle_increment"],
                           config["attention_prior_scale"]).to(self.device)
        self.fuzzy = FuzzyKnowledge(config["scan_angle_min"], config["scan_angle_increment"],
                                    config["v_max"], config["w_max"]).to(self.device)
        self.fusion = HyperFusion(config["v_max"], config["w_max"]).to(self.device)
        self.critic = Critic(state_dim, config["net_width"]).to(self.device)
        self.actor_parameters = tuple(self.actor.parameters()) + tuple(self.fuzzy.parameters()) + tuple(self.fusion.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

    def guidance_floor(self, total_steps):
        return self.config["guidance_floor_initial"] * max(1. - total_steps / self.config["guidance_floor_steps"], 0.)

    def policy_distribution(self, states, floors):
        knowledge = self.fuzzy(states)
        neural, attention = self.actor(states, knowledge["attention_prior"])
        neural_action = self.actor.to_action(neural.mean)
        mean_action, alpha = self.fusion(states, neural_action, knowledge, floors)
        normalized = (mean_action - self.actor.action_bias) / self.actor.action_scale
        hybrid_mean = torch.atanh(normalized.clamp(-1. + 1e-6, 1. - 1e-6))
        mean = torch.where(alpha > 0, hybrid_mean, neural.mean)
        imitation = (alpha.detach() * ((neural_action - knowledge["action"].detach())
                                       / self.fuzzy.action_limits).square()).mean(dim=-1)
        diagnostics = torch.cat((alpha, knowledge["confidence"][:, None]), dim=-1)
        return Normal(mean, neural.stddev), imitation, diagnostics, attention

    @torch.no_grad()
    def select_action(self, state, total_steps, deterministic=False):
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        floor = self.guidance_floor(total_steps)
        distribution, _, diagnostics, _ = self.policy_distribution(states, states.new_full((1,), floor))
        raw_action = distribution.mean if deterministic else distribution.sample()
        action = self.actor.to_action(raw_action)
        log_prob = self.actor.log_prob(distribution, raw_action)
        return (action[0].cpu().numpy(), raw_action[0].cpu().numpy(), log_prob.item(),
                floor, diagnostics[0].cpu().numpy())

    def update(self, buffer):
        batch = buffer.tensors(self.device)
        with torch.no_grad():
            advantages, targets = compute_gae(
                batch["rewards"], self.critic(batch["states"]), self.critic(batch["next_states"]),
                batch["terminated"], batch["truncated"], self.config["gamma"], self.config["gae_lambda"],
            )
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        losses = []
        for _ in range(self.config["epochs"]):
            order = torch.randperm(buffer.size, device=self.device)
            for indices in order.split(self.config["batch_size"]):
                states = batch["states"][indices]
                distribution, imitation, _, _ = self.policy_distribution(states, batch["floors"][indices])
                log_probs = self.actor.log_prob(distribution, batch["raw_actions"][indices])
                ratio = (log_probs - batch["log_probs"][indices]).exp()
                clipped = ratio.clamp(1. - self.config["clip_ratio"], 1. + self.config["clip_ratio"])
                ppo_loss = -torch.minimum(ratio * advantages[indices], clipped * advantages[indices]).mean()
                imitation_loss = self.config["imitation_coef"] * imitation.mean()
                actor_loss = ppo_loss + imitation_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.config["max_grad_norm"])
                self.actor_optimizer.step()

                critic_loss = F.mse_loss(self.critic(states), targets[indices])
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config["max_grad_norm"])
                self.critic_optimizer.step()
                losses.append((ppo_loss.item(), imitation_loss.item(), actor_loss.item()))

        average = np.mean(losses, axis=0)
        guidance = batch["guidance"].mean(dim=0).cpu().tolist()
        result = {"loss": dict(zip(("ppo", "imitation", "total"), map(float, average))),
                  "guidance": dict(linear=guidance[0], angular=guidance[1], confidence=guidance[2],
                                   floor=batch["floors"].mean().item())}
        buffer.clear()
        return result

    def save(self, path, episode, total_steps, updates, recent_outcomes):
        torch.save({
            "algorithm": "PPO_ATTENTION", "version": 1, "observation_shape": (364,),
            "actor": self.actor.state_dict(), "fuzzy": self.fuzzy.state_dict(),
            "fusion": self.fusion.state_dict(), "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config, "episode": episode, "total_steps": total_steps, "updates": updates,
            "recent_outcomes": list(recent_outcomes),
        }, path)

    def load(self, path, training=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if (checkpoint.get("algorithm") != "PPO_ATTENTION" or checkpoint["version"] != 1
                or checkpoint["observation_shape"] != (364,)):
            raise ValueError("A complete PPO_ATTENTION version-1 checkpoint is required; no PPO/KGPPO migration")
        for key in ("net_width", "v_max", "w_max", "scan_angle_min", "scan_angle_increment",
                    "guidance_floor_initial", "guidance_floor_steps", "imitation_coef", "attention_prior_scale"):
            if checkpoint["config"][key] != self.config[key]:
                raise ValueError("Checkpoint {} differs from the current configuration".format(key))
        self.actor.load_state_dict(checkpoint["actor"])
        self.fuzzy.load_state_dict(checkpoint["fuzzy"])
        self.fusion.load_state_dict(checkpoint["fusion"])
        self.critic.load_state_dict(checkpoint["critic"])
        if training:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.config["actor_lr"]
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.config["critic_lr"]
        return {key: checkpoint[key] for key in ("episode", "total_steps", "updates", "recent_outcomes")}


class HighLevelAgent:
    """Plain Dueling DDQN; the Teacher only enters a supervised loss."""
    def __init__(self, config, low_checkpoint, low_source):
        import copy
        from util import SimpleDuelingQNetwork, HighReplayBuffer
        self.config, self.device = dict(config), torch.device(config['device'])
        if (low_checkpoint.get('algorithm') != 'PPO_ATTENTION' or low_checkpoint['version'] != 1
                or low_checkpoint['observation_shape'] != (364,)):
            raise ValueError('A complete PPO_ATTENTION v1 low checkpoint is required')
        self.low_checkpoint, self.low_source = low_checkpoint, dict(low_source)
        self.low = PPOAgent(dict(low_checkpoint['config'],device=config['device']))
        for name in ('actor','fuzzy','fusion','critic'):
            module = getattr(self.low,name)
            module.load_state_dict(low_checkpoint[name])
            module.requires_grad_(False).eval()
        self.velocity_limits = (low_checkpoint['config']['v_max'],low_checkpoint['config']['w_max'])
        self.online = SimpleDuelingQNetwork().to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(),lr=config['high_lr'])
        self.replay = HighReplayBuffer(config['replay_capacity'])
        self.updates = self.learn_rounds = self.target_syncs = 0

    def knowledge_coefficient(self, decisions):
        if self.config['guidance'] == 'off':
            return 0.
        elapsed = max(decisions-self.config['warmup'],0)
        return self.config['lambda_initial']*max(1.-elapsed/self.config['lambda_decay_decisions'],0.)

    def epsilon(self, decisions):
        fraction = min(max(decisions-self.config['warmup'],0)/self.config['epsilon_decay_decisions'],1.)
        return self.config['epsilon_initial']+(self.config['epsilon_final']-self.config['epsilon_initial'])*fraction

    @torch.no_grad()
    def select_action(self, state, mask, decisions, deterministic=False):
        from util import state_tensors
        q = self.online(*state_tensors(state,self.device,self.velocity_limits))[0]
        greedy = int(q.masked_fill(~torch.as_tensor(mask,device=self.device),-torch.inf).argmax().item())
        epsilon = 0. if deterministic else self.epsilon(decisions)
        action = greedy
        if not deterministic and np.random.random() < epsilon:
            action = int(np.random.choice(np.flatnonzero(mask)))
        return action,dict(student_action=action,student_greedy=greedy,epsilon=epsilon)

    def low_action(self, state):
        return self.low.select_action(state,self.low_checkpoint['total_steps'],deterministic=True)[0]

    def update(self, coefficient):
        from util import teacher_cross_entropy
        b = self.replay.sample(self.config['batch_size'],self.device,self.velocity_limits)
        with torch.no_grad():
            targets = b['rewards'].clone()
            bootstrap = ~b['terminated'] & b['next_masks'].any(-1)
            if bootstrap.any():
                states = tuple(x[bootstrap] for x in b['next_states'])
                masks = b['next_masks'][bootstrap]
                actions = self.online(*states).masked_fill(~masks,-torch.inf).argmax(-1,keepdim=True)
                discount = self.config['gamma']**(b['steps'][bootstrap].float()/self.config['discount_reference_steps'])
                targets[bootstrap] += discount*self.target(*states).gather(1,actions).squeeze(1)
        q = self.online(*b['states'])
        td = F.smooth_l1_loss(q.gather(1,b['actions'].long()[:,None]).squeeze(1),targets)
        knowledge = q.new_zeros(())
        if coefficient > 0.:
            knowledge = teacher_cross_entropy(q,b['masks'],b['teacher_actions'],b['teacher_valid'],
                                               self.config['teacher_temperature'],self.config['teacher_q_normalization']=='on')
        loss = td+coefficient*knowledge
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),self.config['max_grad_norm'])
        self.optimizer.step()
        self.updates += 1
        return dict(td=td.item(),knowledge=knowledge.item(),weighted_knowledge=(coefficient*knowledge).item(),total=loss.item())

    def learn(self, decisions):
        coefficient = self.knowledge_coefficient(decisions)
        rows = [self.update(coefficient) for _ in range(self.config['updates_per_decision'])]
        self.learn_rounds += 1
        if self.learn_rounds % self.config['target_sync_decisions'] == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_syncs += 1
        return {key:float(np.mean([row[key] for row in rows])) for key in rows[0]}

    def save(self, path, counters):
        from util import HIGH_SPEC,OnlineOccupancyMap
        torch.save(dict(algorithm='FRONTIER_DDQN',version=1,input_spec=HIGH_SPEC,map_config=OnlineOccupancyMap.config,
                        online=self.online.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
                        replay=self.replay.state_dict(),config=self.config,counters=counters,updates=self.updates,
                        learn_rounds=self.learn_rounds,target_syncs=self.target_syncs,low_checkpoint=self.low_checkpoint,
                        low_source=self.low_source,numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else []),path)

    def restore(self, saved):
        self.online.load_state_dict(saved['online']);self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer']);self.replay.load_state_dict(saved['replay'])
        self.updates,self.learn_rounds,self.target_syncs = saved['updates'],saved['learn_rounds'],saved['target_syncs']
        np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        if self.device.type=='cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
