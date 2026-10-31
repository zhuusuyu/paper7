"""Single-frame PPO and four-frame Dueling Double DQN; no ROS dependencies."""

import copy
import numpy as np
import torch
from torch.nn import functional as F

from util import Actor, Critic, DuelingQNetwork, ReplayBuffer, STEP_BACKEND, TRAINING_VERSION, compute_gae, curriculum_stage


def _check_resume_version(checkpoint, training):
    if training and (checkpoint.get("training_version") != TRAINING_VERSION
                     or checkpoint.get("step_backend") != STEP_BACKEND):
        raise ValueError("Old timing checkpoints are evaluation-only; start a fresh APSR-interaction training run")


class PPOAgent:
    def __init__(self, config, state_dim=364):
        self.config = dict(config)
        self.device = torch.device(config["device"])
        self.actor = Actor(state_dim, config["net_width"], config["v_max"], config["w_max"]).to(self.device)
        self.critic = Critic(state_dim, config["net_width"]).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        distribution = self.actor(states)
        raw_action = distribution.mean if deterministic else distribution.sample()
        action = self.actor.to_action(raw_action)
        log_prob = self.actor.log_prob(distribution, raw_action)
        return action[0].cpu().numpy(), raw_action[0].cpu().numpy(), log_prob.item()

    def update(self, buffer):
        batch = buffer.tensors(self.device)
        with torch.no_grad():
            advantages, targets = compute_gae(
                batch["rewards"], self.critic(batch["states"]), self.critic(batch["next_states"]),
                batch["terminated"], batch["truncated"], self.config["gamma"], self.config["gae_lambda"],
            )
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_losses, critic_losses = [], []
        for _ in range(self.config["epochs"]):
            order = torch.randperm(buffer.size, device=self.device)
            for indices in order.split(self.config["batch_size"]):
                states = batch["states"][indices]
                log_probs = self.actor.log_prob(self.actor(states), batch["raw_actions"][indices])
                ratio = (log_probs - batch["log_probs"][indices]).exp()
                clipped = ratio.clamp(1.0 - self.config["clip_ratio"], 1.0 + self.config["clip_ratio"])
                actor_loss = -torch.minimum(ratio * advantages[indices], clipped * advantages[indices]).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config["max_grad_norm"])
                self.actor_optimizer.step()

                critic_loss = F.mse_loss(self.critic(states), targets[indices])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config["max_grad_norm"])
                self.critic_optimizer.step()
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        buffer.clear()
        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses))}

    def state_dict(self):
        return {
            "training_version": TRAINING_VERSION, "step_backend": STEP_BACKEND,
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
        }

    def load_state_dict(self, checkpoint, training=False):
        for key in ("net_width", "v_max", "w_max", "scan_angle_min", "scan_angle_increment"):
            if checkpoint["config"][key] != self.config[key]:
                raise ValueError("Checkpoint {} differs from the current configuration".format(key))
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if training:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            # Keep CLI learning rates when resuming with a changed training setup.
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.config["actor_lr"]
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.config["critic_lr"]

    def save(self, path, episode, total_steps, updates, recent_outcomes):
        torch.save(dict(self.state_dict(), algorithm="HPPO_LOW", observation_shape=(364,),
                        episode=episode, total_steps=total_steps, updates=updates,
                        recent_outcomes=list(recent_outcomes)), path)

    def load(self, path, training=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        _check_resume_version(checkpoint, training)
        if checkpoint.get("algorithm") != "HPPO_LOW" or checkpoint["observation_shape"] != (364,):
            raise ValueError("A single-frame HPPO low-stage checkpoint is required")
        self.load_state_dict(checkpoint, training)
        counters = {key: checkpoint[key] for key in ("episode", "total_steps", "updates")}
        counters["recent_outcomes"] = checkpoint["recent_outcomes"]
        return counters


class HighLevelAgent:
    """Dueling Double DQN over fixed subgoals, with a frozen PPO actor."""

    def __init__(self, config):
        self.config = dict(config)
        self.device = torch.device(config["device"])
        self.online = DuelingQNetwork(config["net_width"]).to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["high_lr"])
        self.replay = ReplayBuffer(config["replay_capacity"])
        self.updates = self.decisions = 0

    def _load_low_actor(self, actor_state, low_config):
        for key in ("v_max", "w_max", "scan_angle_min", "scan_angle_increment"):
            if low_config[key] != self.config[key]:
                raise ValueError("Low-stage checkpoint {} differs from this environment".format(key))
        self.low_config = dict(low_config)
        self.low_actor = Actor(364, low_config["net_width"], low_config["v_max"], low_config["w_max"]).to(self.device)
        self.low_actor.load_state_dict(actor_state)
        self.low_actor.requires_grad_(False).eval()

    def load_low(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _check_resume_version(checkpoint, True)
        if checkpoint.get("algorithm") != "HPPO_LOW" or checkpoint["observation_shape"] != (364,):
            raise ValueError("--low-checkpoint requires an HPPO low-stage checkpoint")
        self._load_low_actor(checkpoint["actor"], checkpoint["config"])

    def epsilon(self, total_steps):
        fraction = min(total_steps / self.config["epsilon_decay_steps"], 1.0)
        return 1.0 + (self.config["epsilon_final"] - 1.0) * fraction

    @torch.no_grad()
    def select_action(self, state, mask, total_steps, deterministic=False):
        if not deterministic and np.random.random() < self.epsilon(total_steps):
            return int(np.random.choice(np.flatnonzero(mask)))
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid = torch.as_tensor(mask, device=self.device)
        return int(self.online(states).masked_fill(~valid, -torch.inf).argmax(dim=-1).item())

    @torch.no_grad()
    def low_action(self, state):
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.low_actor.to_action(self.low_actor(states).mean)[0].cpu().numpy()

    def update(self):
        batch = self.replay.sample(self.config["batch_size"], self.device)
        with torch.no_grad():
            targets = batch["rewards"].clone()
            bootstrap = ~batch["terminated"]
            if bootstrap.any():
                next_states = batch["next_states"][bootstrap]
                next_actions = self.online(next_states).masked_fill(
                    ~batch["next_masks"][bootstrap], -torch.inf,
                ).argmax(dim=-1, keepdim=True)
                next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
                targets[bootstrap] += self.config["gamma"] * next_values
        values = self.online(batch["states"]).gather(1, batch["actions"][:, None]).squeeze(1)
        loss = F.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config["target_update"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"q_loss": loss.item()}

    def state_dict(self):
        return {
            "training_version": TRAINING_VERSION, "step_backend": STEP_BACKEND,
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
            "low_actor": self.low_actor.state_dict(), "low_config": self.low_config,
            "config": self.config, "updates": self.updates, "decisions": self.decisions,
        }

    def load_state_dict(self, checkpoint, training=False):
        for key in ("net_width", "v_max", "w_max", "scan_angle_min", "scan_angle_increment"):
            if checkpoint["config"][key] != self.config[key]:
                raise ValueError("Checkpoint {} differs from the current configuration".format(key))
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self._load_low_actor(checkpoint["low_actor"], checkpoint["low_config"])
        if training:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for group in self.optimizer.param_groups:
                group["lr"] = self.config["high_lr"]
            self.replay.load_state_dict(checkpoint["replay"])
        self.updates, self.decisions = checkpoint["updates"], checkpoint["decisions"]

    def save(self, path, episode, total_steps, updates, recent_outcomes):
        torch.save(dict(self.state_dict(), algorithm="HPPO_HIGH", observation_shape=(4, 364),
                        episode=episode, total_steps=total_steps, updates=updates,
                        recent_outcomes=list(recent_outcomes)), path)

    def load(self, path, training=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _check_resume_version(checkpoint, training)
        if checkpoint.get("algorithm") != "HPPO_HIGH" or checkpoint["observation_shape"] != (4, 364):
            raise ValueError("A four-frame HPPO high-stage checkpoint is required")
        self.load_state_dict(checkpoint, training)
        return {key: checkpoint[key] for key in ("episode", "total_steps", "updates", "recent_outcomes")}


class AutoAgent:
    """Own both learners and checkpoint their shared interaction budget."""

    def __init__(self, config):
        self.config = dict(config)
        self.low = PPOAgent(config)
        self.high = HighLevelAgent(config)
        self.high.low_actor = copy.deepcopy(self.low.actor).requires_grad_(False).eval()
        self.high.low_config = dict(config)
        self.phase = "low"
        self.low_steps = self.high_steps = self.low_updates = 0
        self.pretrain_limit = config["max_train_steps"] // 2
        self.switch_step = None
        self.gate_result = None
        self.status = "training"

    def sync_low(self):
        self.high.low_actor.load_state_dict(self.low.actor.state_dict())

    def enter_high(self):
        self.sync_low()
        self.low.actor.requires_grad_(False)
        self.low.critic.requires_grad_(False)
        self.phase = "high"
        self.switch_step = self.low_steps

    def save(self, path, episode, total_steps, updates, recent_outcomes):
        if self.phase == "low":
            self.sync_low()
        torch.save({
            "training_version": TRAINING_VERSION, "step_backend": STEP_BACKEND,
            "algorithm": "HPPO_AUTO", "observation_shape": (4, 364), "config": self.config,
            "phase": self.phase, "low_steps": self.low_steps, "high_steps": self.high_steps,
            "low_updates": self.low_updates, "low": self.low.state_dict(), "high": self.high.state_dict(),
            "episode": episode, "total_steps": total_steps, "updates": updates,
            "recent_outcomes": list(recent_outcomes),
            "pretrain_limit": self.pretrain_limit, "switch_step": self.switch_step,
            "gate_result": self.gate_result, "status": self.status,
            "curriculum_stage": curriculum_stage(self.low_steps),
        }, path)

    def load(self, path, training=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _check_resume_version(checkpoint, training)
        if checkpoint.get("algorithm") != "HPPO_AUTO" or checkpoint["observation_shape"] != (4, 364):
            raise ValueError("An HPPO_AUTO checkpoint is required; use --stage low/high for standalone models")
        if training and checkpoint["config"]["pretrain_steps"] != self.config["pretrain_steps"]:
            raise ValueError("--pretrain-steps must match the automatic checkpoint")
        total_steps = checkpoint["total_steps"]
        low_steps, high_steps, phase = checkpoint["low_steps"], checkpoint["high_steps"], checkpoint["phase"]
        if checkpoint.get("training_version") in (2, TRAINING_VERSION):
            self.pretrain_limit, self.switch_step = checkpoint["pretrain_limit"], checkpoint["switch_step"]
            self.gate_result, self.status = checkpoint["gate_result"], checkpoint["status"]
        else:
            # Explicit policy-only compatibility with the old fixed handoff.
            self.switch_step = low_steps if phase == "high" else None
            self.status = "legacy_evaluation"
        if (low_steps + high_steps != total_steps or phase not in ("low", "high")
                or (phase == "low" and (high_steps != 0 or self.switch_step is not None))
                or (phase == "high" and self.switch_step != low_steps)):
            raise ValueError("Automatic checkpoint phase and step counters are inconsistent")
        if training and not self.config["pretrain_steps"] <= self.pretrain_limit < self.config["max_train_steps"]:
            raise ValueError("Restored pretraining limit is incompatible with the training budget")
        if training and self.status == "low_not_qualified":
            raise ValueError("This run stopped at its low-level qualification limit; start a new training run")
        if training:
            self.status = "training"
        self.low.load_state_dict(checkpoint["low"], training)
        self.high.load_state_dict(checkpoint["high"], training)
        self.phase = phase
        self.low_steps, self.high_steps, self.low_updates = low_steps, high_steps, checkpoint["low_updates"]
        if phase == "high":
            self.low.actor.requires_grad_(False)
            self.low.critic.requires_grad_(False)
        return {key: checkpoint[key] for key in ("episode", "total_steps", "updates", "recent_outcomes")}
