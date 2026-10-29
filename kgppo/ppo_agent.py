"""PPO-Clip with trainable fuzzy guidance and a shared squashed Gaussian policy."""

import math
import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import functional as F

from util import Actor, Critic, FuzzyActor, compute_gae


class KGPPOAgent:
    def __init__(self, config, state_dim=364):
        self.config = dict(config)
        self.device = torch.device(config["device"])
        self.actor = Actor(state_dim, config["net_width"], config["v_max"], config["w_max"]).to(self.device)
        self.critic = Critic(state_dim, config["net_width"]).to(self.device)
        self.fuzzy_actor = FuzzyActor(
            config["scan_angle_min"], config["scan_angle_increment"], config["v_max"], config["w_max"],
        ).to(self.device)
        self.actor_parameters = tuple(self.actor.parameters()) + tuple(self.fuzzy_actor.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])

    def guidance_weight(self, total_steps):
        fraction = total_steps / self.config["guidance_steps"]
        if fraction >= 1.0:
            return 0.0
        rate = self.config["guidance_exp_rate"]
        return (self.config["guidance_initial"] * math.exp(-rate * fraction)
                * math.expm1(-rate * (1.0 - fraction)) / math.expm1(-rate))

    def policy_distribution(self, states, alphas):
        neural = self.actor(states)
        imitation_error = states.new_zeros(states.shape[0])
        if torch.any(alphas > 0):
            neural_action = self.actor.to_action(neural.mean)
            fuzzy_action = self.fuzzy_actor(states)
            mean_action = (1.0 - alphas[:, None]) * neural_action + alphas[:, None] * fuzzy_action
            normalized = (mean_action - self.actor.action_bias) / self.actor.action_scale
            hybrid_mean = torch.atanh(normalized.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
            mean = torch.where(alphas[:, None] > 0, hybrid_mean, neural.mean)
            distribution = Normal(mean, neural.stddev)
            imitation_error = (
                (neural_action - fuzzy_action.detach()) / self.fuzzy_actor.action_limits
            ).square().mean(dim=-1)
        else:
            distribution = neural
        return distribution, imitation_error

    @torch.no_grad()
    def select_action(self, state, total_steps, deterministic=False):
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        alpha = self.guidance_weight(total_steps)
        distribution, _ = self.policy_distribution(states, states.new_full((1,), alpha))
        raw_action = distribution.mean if deterministic else distribution.sample()
        action = self.actor.to_action(raw_action)
        log_prob = self.actor.log_prob(distribution, raw_action)
        return action[0].cpu().numpy(), raw_action[0].cpu().numpy(), log_prob.item(), alpha

    def update(self, buffer):
        batch = buffer.tensors(self.device)
        with torch.no_grad():
            advantages, targets = compute_gae(
                batch["rewards"], self.critic(batch["states"]), self.critic(batch["next_states"]),
                batch["terminated"], batch["truncated"], self.config["gamma"], self.config["gae_lambda"],
            )
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_losses, critic_losses, imitation_losses = [], [], []
        for _ in range(self.config["epochs"]):
            order = torch.randperm(buffer.size, device=self.device)
            for indices in order.split(self.config["batch_size"]):
                states = batch["states"][indices]
                alphas = batch["alphas"][indices]
                distribution, imitation_error = self.policy_distribution(states, alphas)
                log_probs = self.actor.log_prob(distribution, batch["raw_actions"][indices])
                ratio = (log_probs - batch["log_probs"][indices]).exp()
                clipped = ratio.clamp(1.0 - self.config["clip_ratio"], 1.0 + self.config["clip_ratio"])
                imitation_loss = (alphas * imitation_error).mean()
                actor_loss = -torch.minimum(ratio * advantages[indices], clipped * advantages[indices]).mean() + imitation_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.config["max_grad_norm"])
                self.actor_optimizer.step()

                critic_loss = F.mse_loss(self.critic(states), targets[indices])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config["max_grad_norm"])
                self.critic_optimizer.step()
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                imitation_losses.append(imitation_loss.item())

        buffer.clear()
        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses)),
                "imitation_loss": float(np.mean(imitation_losses))}

    def save(self, path, episode, total_steps, updates, recent_outcomes):
        torch.save({
            "algorithm": "KGPPO",
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "fuzzy_actor": self.fuzzy_actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config, "episode": episode, "total_steps": total_steps, "updates": updates,
            "recent_outcomes": list(recent_outcomes),
        }, path)

    def load(self, path, training=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != "KGPPO":
            raise ValueError("A complete KGPPO checkpoint is required; baseline PPO checkpoints cannot be resumed")
        for key in ("net_width", "v_max", "w_max", "guidance_initial", "guidance_steps", "guidance_exp_rate",
                    "scan_angle_min", "scan_angle_increment"):
            if checkpoint["config"][key] != self.config[key]:
                raise ValueError("Checkpoint {} differs from the current configuration".format(key))
        self.actor.load_state_dict(checkpoint["actor"])
        self.fuzzy_actor.load_state_dict(checkpoint["fuzzy_actor"])
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
        counters["recent_outcomes"] = checkpoint["recent_outcomes"]
        return counters
