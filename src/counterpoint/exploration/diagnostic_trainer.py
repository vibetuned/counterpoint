"""
Diagnostic Trainer for debugging action masking and learning.

Logs detailed statistics about:
- Finger usage distribution
- Action mask statistics
- Reward breakdown
- Episode statistics
"""

import torch
import tqdm
import sys
from skrl.trainers.torch import SequentialTrainer


class DiagnosticTrainer(SequentialTrainer):
    """
    Custom trainer with detailed diagnostic logging for TensorBoard.
    
    Extends SKRL's SequentialTrainer to add logging of:
    1. Finger selection distribution (which fingers are being used)
    2. Action mask statistics (how restrictive is the mask)
    3. Reward statistics per episode
    4. Priority head behavior
    """
    
    def __init__(self, cfg, env, agents):
        super().__init__(cfg=cfg, env=env, agents=agents)
        self._log_interval = 10  # Log detailed stats every N rollouts
        self._rollout_count = 0
        
        # Accumulators for statistics
        self._finger_counts = torch.zeros(5)  # Count of each finger used
        self._action_count = 0
        self._rewards_sum = 0.0
        self._rewards_count = 0
        self._no_finger_count = 0  # Count of actions with no finger pressed
        self._mask_forbidden_sum = 0.0  # Sum of forbidden fingers per step
        self._episode_rewards = []
        
        # Tau annealing params
        exp_cfg = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
        self.tau_start = exp_cfg.get("tau_start", 5.0)
        self.tau_end = exp_cfg.get("tau_end", 0.5)
        self.tau_fraction = exp_cfg.get("tau_fraction", 0.5)
    
    def train(self):
        """Override train to add diagnostic logging."""
        
        # Set training mode
        self.agents.set_running_mode("train")
        
        # Reset environments
        observations, infos = self.env.reset()
        episode_reward = 0.0
        
        # Get timesteps config
        timesteps = self.cfg.get("timesteps", self.cfg["timesteps"]) if isinstance(self.cfg, dict) else self.cfg.timesteps
        rollouts = self.cfg.get("rollouts", 1024) if isinstance(self.cfg, dict) else self.cfg.rollouts
        
        for timestep in tqdm.tqdm(range(timesteps), disable=False, file=sys.stdout):
            
            # Pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=timesteps)
            
            with torch.no_grad():
                # Compute actions
                act_result = self.agents.act(
                    observations, 
                    timestep=timestep, 
                    timesteps=timesteps
                )
                actions = act_result[0]
                
                # Log action statistics
                self._log_action_stats(actions, observations)
                
                # Step the environments
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                
                # Track rewards
                episode_reward += rewards.sum().item()
                self._rewards_sum += rewards.sum().item()
                self._rewards_count += rewards.numel()
                
                # Record transition
                self.agents.record_transition(
                    states=observations,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_observations,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps
                )
            
            # Post-interaction (triggers learning updates)
            self.agents.post_interaction(timestep=timestep, timesteps=timesteps)
            
            # Log accumulated stats every rollout
            if (timestep + 1) % rollouts == 0:
                self._rollout_count += 1
                self._log_accumulated_stats()
                #self._update_tau(timestep, timesteps)
                self._reset_accumulators()
            
            # Handle episode end
            if terminated.any() or truncated.any():
                self._episode_rewards.append(episode_reward)
                episode_reward = 0.0
                observations, infos = self.env.reset()
            else:
                observations = next_observations
        
        # Close environment
        self.env.close()
    
    def _log_action_stats(self, actions, observations):
        """Log statistics about the actions taken."""
        # actions: (batch, 10) - [finger1..5, black1..5]
        batch_size = actions.shape[0]
        self._action_count += batch_size
        
        # Extract finger selections (first 5 actions)
        fingers = actions[:, :5]  # (batch, 5)
        
        # Count finger usage
        for i in range(5):
            self._finger_counts[i] += (fingers[:, i] == 1).sum().item()
        
        # Count actions with no finger pressed
        no_finger = (fingers.sum(dim=1) == 0).sum().item()
        self._no_finger_count += no_finger
        
        # Extract action mask from observations
        # observations: (batch, 1053) = [mask(11), grid(1040), hand_state(1), relative_target(1)]
        if observations.shape[1] >= 11:
            action_mask = observations[:, :11]
            finger_mask = action_mask[:, 6:11]  # (batch, 5) - 1=allowed, 0=forbidden
            
            # Count forbidden fingers per step
            forbidden_count = (finger_mask == 0).sum(dim=1).float().mean().item()
            self._mask_forbidden_sum += forbidden_count
    
    def _log_accumulated_stats(self):
        """Log accumulated statistics to TensorBoard."""
        if self._action_count == 0:
            return
            
        # Finger usage distribution (normalized to percentages)
        total_fingers = self._finger_counts.sum().item()
        if total_fingers > 0:
            for i in range(5):
                pct = 100.0 * self._finger_counts[i].item() / total_fingers
                self.agents.track_data(f"Diagnostics / Finger {i+1} Usage %", pct)
        
        # No-finger rate
        no_finger_rate = 100.0 * self._no_finger_count / self._action_count
        self.agents.track_data("Diagnostics / No Finger Rate %", no_finger_rate)
        
        # Average fingers per action
        avg_fingers = total_fingers / self._action_count if self._action_count > 0 else 0
        self.agents.track_data("Diagnostics / Avg Fingers per Action", avg_fingers)
        
        # Average forbidden fingers from mask
        avg_forbidden = self._mask_forbidden_sum / self._action_count if self._action_count > 0 else 0
        self.agents.track_data("Diagnostics / Avg Forbidden Fingers per Step", avg_forbidden)
        
        # Reward statistics
        if self._rewards_count > 0:
            avg_reward = self._rewards_sum / self._rewards_count
            self.agents.track_data("Diagnostics / Avg Step Reward", avg_reward)
        
        # Episode rewards
        if self._episode_rewards:
            avg_ep_reward = sum(self._episode_rewards) / len(self._episode_rewards)
            self.agents.track_data("Diagnostics / Avg Episode Reward", avg_ep_reward)
    
    def _reset_accumulators(self):
        """Reset statistics accumulators."""
        self._finger_counts.zero_()
        self._action_count = 0
        self._rewards_sum = 0.0
        self._rewards_count = 0
        self._no_finger_count = 0
        self._mask_forbidden_sum = 0.0
        self._episode_rewards = []
    
    def _update_tau(self, timestep, total_timesteps):
        """Update Gumbel-Softmax temperature."""
        progress = timestep / total_timesteps
        
        if progress < self.tau_fraction:
            frac = progress / self.tau_fraction
            current_tau = self.tau_start - frac * (self.tau_start - self.tau_end)
        else:
            current_tau = self.tau_end
            
        policy = None
        if hasattr(self.agents, "policy"):
            policy = self.agents.policy
        elif hasattr(self.agents, "models") and "policy" in self.agents.models:
            policy = self.agents.models["policy"]
            
        if policy and hasattr(policy, "set_tau"):
            policy.set_tau(current_tau)
            if self._rollout_count % self._log_interval == 0:
                self.agents.track_data("Diagnostics / Tau", current_tau)
