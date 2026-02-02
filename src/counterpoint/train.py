import gymnasium as gym
import torch
import torch.nn as nn
import os
import yaml
import counterpoint.envs # Register envs

# Import SKRL components
from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from counterpoint.policies import (
    FlattenActionWrapper,
    SimpleBaselinePolicy, SimpleBaselineValue,
    ConvPolicy, ConvValue,
    TransformerPolicy, TransformerValue,
)

# Policy registry
POLICY_REGISTRY = {
    "simple": (SimpleBaselinePolicy, SimpleBaselineValue),
    "conv": (ConvPolicy, ConvValue),
    "transformer": (TransformerPolicy, TransformerValue),
}

def get_models(policy_type, observation_space, action_space, device):
    """Get policy and value models based on architecture type."""
    if policy_type not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy type: {policy_type}. Available: {list(POLICY_REGISTRY.keys())}")
    
    policy_cls, value_cls = POLICY_REGISTRY[policy_type]
    return {
        "policy": policy_cls(observation_space, action_space, device),
        "value": value_cls(observation_space, action_space, device),
    }

def load_config(path="conf/base.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train(steps=None, resume_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Config
    try:
        config = load_config()
    except FileNotFoundError:
        # Fallback or error?
        print("Config file conf/base.yaml not found!")
        return

    # Override steps if provided
    current_timesteps = config["training"]["timesteps"]
    if steps is not None:
        print(f"Overriding timesteps: {current_timesteps} -> {steps}")
        config["training"]["timesteps"] = steps
        
    # Load Env
    env_name = config["env"]["name"]
    env = gym.make(env_name)
    env = FlattenActionWrapper(env)
    env = wrap_env(env) # SKRL wrapper

    # Define Agent - select architecture from config
    policy_type = config.get("policy", {}).get("type", "simple")
    print(f"Using policy architecture: {policy_type}")
    models = get_models(policy_type, env.observation_space, env.action_space, device)

    # Memory
    memory = RandomMemory(memory_size=config["training"]["rollouts"], num_envs=env.num_envs, device=device)

    # PPO Config
    # Map from our yaml to SKRL structure
    # SKRL PPO config structure is specific. We'll reconstruct it or update default.
    from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
    cfg_agent = PPO_DEFAULT_CONFIG.copy()
    
    # Map Training params
    train_cfg = config["training"]
    cfg_agent["rollouts"] = train_cfg["rollouts"]
    cfg_agent["learning_epochs"] = train_cfg["learning_epochs"]
    cfg_agent["mini_batches"] = train_cfg["mini_batches"]
    cfg_agent["discount_factor"] = train_cfg["discount_factor"]
    cfg_agent["lambda"] = train_cfg["lambda_gae"]
    cfg_agent["learning_rate"] = float(train_cfg["learning_rate"])
    cfg_agent["grad_norm_clip"] = train_cfg["grad_norm_clip"]
    cfg_agent["ratio_clip"] = train_cfg["ratio_clip"]
    cfg_agent["value_loss_scale"] = train_cfg["value_loss_scale"]
    cfg_agent["entropy_loss_scale"] = train_cfg["entropy_loss_scale"]
    
    # Logging
    exp_cfg = config["experiment"]
    cfg_agent["experiment"]["write_interval"] = exp_cfg["write_interval"]
    cfg_agent["experiment"]["checkpoint_interval"] = exp_cfg["checkpoint_interval"]
    cfg_agent["experiment"]["directory"] = exp_cfg["directory"]
    
    # Timesteps
    cfg_agent["timesteps"] = config["training"]["timesteps"]

    agent = PPO(models=models, 
                memory=memory, 
                cfg=cfg_agent, 
                observation_space=env.observation_space, 
                action_space=env.action_space, 
                device=device)

    # Resume if requested
    if resume_path:
        print(f"Resuming training from checkpoint: {resume_path}")
        agent.load(resume_path)

    # Trainer
    trainer = SequentialTrainer(cfg=cfg_agent, env=env, agents=agent)

    # Train
    print(f"Starting training for {cfg_agent['timesteps']} steps...")
    trainer.train()

def test(path=None):
    import time
    import glob
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Config (for Env params mostly)
    config = load_config()

    # Load Env with Render
    env_name = config["env"]["name"]
    env = gym.make(env_name, render_mode="human") 
    env = FlattenActionWrapper(env)
    env = wrap_env(env)

    # Define Agent - select architecture from config
    policy_type = config.get("policy", {}).get("type", "simple")
    print(f"Using policy architecture: {policy_type}")
    models = get_models(policy_type, env.observation_space, env.action_space, device)

    from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
    cfg_agent = PPO_DEFAULT_CONFIG.copy()
    exp_cfg = config["experiment"]
    cfg_agent["experiment"]["directory"] = exp_cfg["directory"]

    agent = PPO(models=models, 
                memory=None, 
                cfg=cfg_agent, 
                observation_space=env.observation_space, 
                action_space=env.action_space, 
                device=device)

    # Find Checkpoint if not provided
    if path is None:
        runs_dir = exp_cfg["directory"]
        if not os.path.exists(runs_dir):
            print(f"No runs found in {runs_dir}")
            return
            
        subdirs = glob.glob(os.path.join(runs_dir, "*"))
        if not subdirs:
            print("No run subdirectories found.")
            return
            
        # exclude non-dir files?
        subdirs = [d for d in subdirs if os.path.isdir(d)]
        if not subdirs:
             print("No run subdirectories found.")
             return

        latest_subdir = max(subdirs, key=os.path.getmtime)
        print(f"Using latest run: {latest_subdir}")
        
        ckpts = glob.glob(os.path.join(latest_subdir, "checkpoints", "*.pt"))
        if not ckpts:
            print("No checkpoints found in latest run.")
            return
            
        path = max(ckpts, key=os.path.getmtime)
    
    print(f"Loading checkpoint: {path}")
    agent.load(path)
    
    print("Starting inference... (Press Ctrl+C to stop)")
    try:
        states, _ = env.reset()
        while True:
            with torch.no_grad():
                actions = agent.act(states, timestep=0, timesteps=0)[0]
            
            states, rewards, terminated, truncated, infos = env.step(actions)
            
            time.sleep(0.5) 
            
            if terminated.any() or truncated.any():
                states, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping...")
        env.close()

if __name__ == "__main__":
    train()
