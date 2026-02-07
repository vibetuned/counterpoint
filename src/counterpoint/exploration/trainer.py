
import torch
import tqdm
import sys
from skrl.trainers.torch import SequentialTrainer

class ExplorationTrainer(SequentialTrainer):
    """
    Custom trainer that integrates BC loss and RND intrinsic rewards.
    
    Extends SKRL's SequentialTrainer to add:
    1. BC loss as auxiliary loss after each training epoch
    2. RND intrinsic rewards added to environment rewards
    3. BC coefficient decay over training
    """
    
    def __init__(self, cfg, env, agents, exploration_manager):
        super().__init__(cfg=cfg, env=env, agents=agents)
        self.exploration_manager = exploration_manager
        self._epoch = 0
        self._log_interval = 10  # Log stats every N epochs
        
        # Tau annealing params (defaults)
        # We can try to read from cfg["experiment"] if available
        exp_cfg = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
        self.tau_start = exp_cfg.get("tau_start", 5.0)
        self.tau_end = exp_cfg.get("tau_end", 0.5)
        self.tau_fraction = exp_cfg.get("tau_fraction", 0.5)
    
    def train(self):
        """
        Override train to add exploration mechanisms.
        
        Follows SKRL's SequentialTrainer pattern with added BC/RND support.
        """
        
        # Set training mode
        self.agents.set_running_mode("train")
        
        # Reset environments
        observations, infos = self.env.reset()
        
        # Get timesteps config
        timesteps = self.cfg.get("timesteps", self.cfg["timesteps"]) if isinstance(self.cfg, dict) else self.cfg.timesteps
        
        for timestep in tqdm.tqdm(range(timesteps), disable=False, file=sys.stdout):
            
            # Pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=timesteps)
            
            with torch.no_grad():
                # Compute actions (act returns (actions, outputs, ...))
                act_result = self.agents.act(
                    observations, 
                    timestep=timestep, 
                    timesteps=timesteps
                )
                actions = act_result[0]
                
                # Step the environments
                next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                
                # Add RND intrinsic reward if enabled
                if self.exploration_manager.use_rnd:
                    intrinsic = self.exploration_manager.compute_intrinsic_reward(observations)
                    rewards = rewards + intrinsic.unsqueeze(-1)
                    # Track intrinsic reward
                    self.agents.track_data("Exploration / RND intrinsic reward", intrinsic.mean().item())
                
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
                
                # Log environment info
                if "episode" in infos:
                    for k, v in infos["episode"].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())
            
            # Post-interaction (triggers learning updates)
            self.agents.post_interaction(timestep=timestep, timesteps=timesteps)
            
            # Apply BC loss after PPO update (every rollout cycle)
            rollouts = self.cfg.get("rollouts", 1024) if isinstance(self.cfg, dict) else self.cfg.rollouts
            if (timestep + 1) % rollouts == 0:
                self._apply_bc_loss()
                self._update_rnd()
                self._update_tau(timestep, timesteps)
                
                # Decay BC coefficient
                self.exploration_manager.decay_bc()
                self._epoch += 1
                
                # Log exploration stats
                if self._epoch % self._log_interval == 0:
                    stats = self.exploration_manager.get_stats()
                    if stats:
                        for k, v in stats.items():
                            self.agents.track_data(k, v)
            
            # Reset environments
            if terminated.any() or truncated.any():
                observations, infos = self.env.reset()
            else:
                observations = next_observations
        
        # Close environment
        self.env.close()
    
    def _apply_bc_loss(self):
        """Apply behavior cloning loss as auxiliary update."""
        if not self.exploration_manager.use_bc:
            return
        
        bc_loss = self.exploration_manager.compute_bc_loss(
            self.agents.models["policy"],
            batch_size=64
        )
        
        if bc_loss.item() > 0:
            self.agents.optimizer.zero_grad()
            bc_loss.backward()
            self.agents.optimizer.step()
            # Track BC loss
            self.agents.track_data("Exploration / BC loss", bc_loss.item())
    
    def _update_rnd(self):
        """Update RND predictor network with recent observations."""
        if not self.exploration_manager.use_rnd:
            return
        
        try:
            samples = self.agents.memory.sample(
                names=["states"], 
                batch_size=min(256, self.cfg.get("rollouts", 1024) if isinstance(self.cfg, dict) else self.cfg.rollouts)
            )
            if samples is not None and "states" in samples:
                self.exploration_manager.update_rnd(samples["states"])
        except Exception:
            pass  # Memory might not have enough samples yet
            
    def _update_tau(self, timestep, total_timesteps):
        """Update Gumbel-Softmax temperature."""
        # Check if policy supports set_tau
        if not hasattr(self.agents, "policy") and not hasattr(self.agents, "models"):
             return

        # Handle annealing
        # Progress: 0.0 -> 1.0
        progress = timestep / total_timesteps
        
        if progress < self.tau_fraction:
            # Linear decay
            frac = progress / self.tau_fraction
            current_tau = self.tau_start - frac * (self.tau_start - self.tau_end)
        else:
            current_tau = self.tau_end
            
        # Update policy
        # SKRL PPO agent: policy is in self.agents.policy (if exposed) or self.agents.models["policy"]
        policy = None
        if hasattr(self.agents, "policy"):
            policy = self.agents.policy
        elif hasattr(self.agents, "models") and "policy" in self.agents.models:
            policy = self.agents.models["policy"]
            
        if policy and hasattr(policy, "set_tau"):
            policy.set_tau(current_tau)
            # Log tau occasionally
            if self._epoch % self._log_interval == 0:
                self.agents.track_data("Exploration / Tau", current_tau)
