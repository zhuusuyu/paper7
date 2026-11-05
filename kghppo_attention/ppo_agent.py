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
    """Knowledge-guided Double DQN over a complete frozen low policy."""

    def __init__(self, config, low_checkpoint):
        import copy
        from util import GuidedHighQNetwork, HighReplay

        self.config = dict(config)
        self.device = torch.device(config['device'])
        if (low_checkpoint.get('algorithm') != 'PPO_ATTENTION' or low_checkpoint['version'] != 1
                or low_checkpoint['observation_shape'] != (364,)):
            raise ValueError('Low policy must be a complete PPO_ATTENTION v1 checkpoint')
        self.low_checkpoint = low_checkpoint
        self.low = PPOAgent(dict(low_checkpoint['config'], device=config['device']))
        for name in ('actor', 'fuzzy', 'fusion', 'critic'):
            module = getattr(self.low, name)
            module.load_state_dict(low_checkpoint[name])
            module.requires_grad_(False).eval()
        self.online = GuidedHighQNetwork(config['net_width'], low_checkpoint['config']).to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config['high_lr'])
        self.replay = HighReplay(config['replay_capacity'])
        self.updates = 0

    def guidance_floor(self, total_steps):
        return self.config['high_guidance_floor_initial'] * max(1. - total_steps / self.config['high_guidance_floor_steps'], 0.)

    def select_action(self, state, mask, total_steps, deterministic=False):
        epsilon = 1 + (.05 - 1) * min(total_steps / self.config['epsilon_decay_steps'], 1)
        with torch.no_grad():
            states = torch.as_tensor(state, device=self.device).unsqueeze(0)
            masks = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            output = self.online(states, masks, states.new_full((1,), self.guidance_floor(total_steps)))
        if not deterministic and np.random.random() < epsilon:
            probabilities = output['exploration'][0].cpu().numpy().astype(np.float64)
            return int(np.random.choice(225, p=probabilities / probabilities.sum()))
        return int(output['q'].masked_fill(~masks, -torch.inf).argmax(-1).item())

    def low_action(self, state):
        return self.low.select_action(state, self.low_checkpoint['total_steps'], deterministic=True)[0]

    def update(self):
        batch = self.replay.sample(self.config['batch_size'], self.device)
        with torch.no_grad():
            targets = batch['rewards'].clone()
            bootstrap = ~batch['terminated']
            if bootstrap.any():
                states = batch['next_states'][bootstrap]
                masks, floors = batch['next_masks'][bootstrap], batch['next_floors'][bootstrap]
                actions = self.online(states, masks, floors)['q'].masked_fill(~masks, -torch.inf).argmax(-1, keepdim=True)
                targets[bootstrap] += self.config['gamma'] * self.target(states, masks, floors)['q'].gather(1, actions).squeeze(1)
        output = self.online(batch['states'], batch['masks'], batch['floors'])
        td_loss = F.smooth_l1_loss(output['q'].gather(1, batch['actions'][:, None]).squeeze(1), targets)
        imitation_loss = self.config['high_imitation_coef'] * output['imitation'].mean()
        loss = td_loss + imitation_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config['max_grad_norm'])
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config['target_update'] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return dict(loss=dict(td=td_loss.item(), imitation=imitation_loss.item(), total=loss.item()),
                    guidance=dict(floor=batch['floors'].mean().item(), alpha=output['alpha'].mean().item(),
                                  preference=output['preference'].mean().item(), confidence=output['confidence'].mean().item()))

    def save(self, path, counters):
        torch.save(dict(algorithm='KGHPPO_ATTENTION', version=1, observation_shape=(1456,),
                        config=self.config, low_checkpoint=self.low_checkpoint,
                        online=self.online.state_dict(), target=self.target.state_dict(),
                        optimizer=self.optimizer.state_dict(), replay=self.replay.state_dict(),
                        updates=self.updates, counters=counters, numpy_rng=np.random.get_state(),
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else []), path)

    def restore(self, saved):
        self.online.load_state_dict(saved['online'])
        self.target.load_state_dict(saved['target'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.replay.load_state_dict(saved['replay'])
        self.updates = saved['updates']
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if self.device.type == 'cuda' and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
