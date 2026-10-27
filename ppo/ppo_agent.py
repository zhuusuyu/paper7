"""PPO-Clip with a squashed Gaussian policy; no ROS dependencies."""

import numpy as np
import torch
from torch.nn import functional as F

from util import Actor, Critic, compute_gae


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

    def save(self, path, episode, total_steps, updates, recent_outcomes):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config, "episode": episode, "total_steps": total_steps, "updates": updates,
            "recent_outcomes": list(recent_outcomes),
        }, path)

    def load(self, path, training=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for key in ("net_width", "v_max", "w_max"):
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
        counters = {key: checkpoint[key] for key in ("episode", "total_steps", "updates")}
        counters["recent_outcomes"] = checkpoint.get("recent_outcomes", [])
        return counters
