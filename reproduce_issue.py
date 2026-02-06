
import gymnasium as gym
import numpy as np
import torch
import counterpoint.envs
from counterpoint.policies import FlattenActionWrapper
from counterpoint.policies.transformer import TransformerPolicy, TransformerValue
from counterpoint.policies.cnn import ConvPolicy, ConvValue
from counterpoint.policies.simple import SimpleBaselinePolicy, SimpleBaselineValue
from counterpoint.policies.decoder import DecoderPolicy, DecoderValue
from counterpoint.data.demonstrations import DemonstrationGenerator
from counterpoint.exploration.rnd import RNDNetwork
from skrl.envs.wrappers.torch import wrap_env

def check_env_and_policies():
    env_name = "Piano-v0" 
    print(f"Creating env: {env_name}")
    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Failed to make env: {e}")
        return

    # Apply wrappers 
    env = FlattenActionWrapper(env)
    env = wrap_env(env)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    obs, _ = env.reset()
    print("Obs Shape:", obs.shape)
    
    inputs = {"states": obs}
    
    # Test all policies
    policies = [
        ("Transformer", TransformerPolicy, TransformerValue),
        ("CNN", ConvPolicy, ConvValue),
        ("Simple", SimpleBaselinePolicy, SimpleBaselineValue),
        ("Decoder", DecoderPolicy, DecoderValue)
    ]
    
    for name, P, V in policies:
        print(f"\nTesting {name} Policy...")
        try:
            policy = P(env.observation_space, env.action_space, device)
            policy.to(device)
            policy.act(inputs, role="policy")
            print(f"{name} Policy OK")
            
            value = V(env.observation_space, env.action_space, device)
            value.to(device)
            value.act(inputs, role="value")
            print(f"{name} Value OK")
        except Exception as e:
            print(f"{name} Failed: {e}")
            import traceback
            traceback.print_exc()

def check_bc_generator():
    print("\nChecking BC Demonstration Generator...")
    try:
        gen = DemonstrationGenerator()
        obs, actions = gen.sample_batch_flat(batch_size=4)
        
        print("BC Obs Shape:", obs.shape)
        print("BC Actions Shape:", actions.shape)
        
        expected_dim = 1048
        if obs.shape[1] == expected_dim:
            print("BC Obs Dimension OK")
            # Check mask placement (first 6 elements)
            
            first_obs = obs[0]
            mask = first_obs[:6]
            print("Sample Mask:", mask)
            
            if mask[-1] == 1.0:
                 print("Mask num_notes check OK (1.0)")
            else:
                 print(f"Mask num_notes check FAILED (expected 1.0, got {mask[-1]})")
                 
        else:
             print(f"BC Obs Dimension FAILED (expected {expected_dim}, got {obs.shape[1]})")

    except Exception as e:
        print(f"BC Check Failed: {e}")
        import traceback
        traceback.print_exc()

def check_rnd(device):
    print("\nChecking RND Network...")
    try:
        # Obs dim = 1048 (including mask)
        rnd = RNDNetwork(obs_dim=1048).to(device)
        
        # Create dummy observation with REALISTIC magnitudes
        # Hand pos (index 1046) ~ 40-50
        # Relative target (index 1047) ~ -20 to 20
        # Grid (index 6-1045) ~ 0 or 1
        batch_size = 1
        obs = torch.zeros(batch_size, 1048).to(device)
        obs[:, 6:1046] = torch.randint(0, 2, (batch_size, 1040)).float() # Grid
        obs[:, 1046] = 50.0 # Hand state
        obs[:, 1047] = 10.0 # Relative target
        
        # 0-5 mask (randomly set)
        obs[:, :5] = 0.0
        obs[:, 5] = 1.0 # num notes
        
        # Forward pass
        target, predictor = rnd(obs)
        print("RND Forward Pass OK")
        print("Target Shape:", target.shape)
        print("Target Mean:", target.mean().item(), "Std:", target.std().item())
        print("Predictor Mean:", predictor.mean().item(), "Std:", predictor.std().item())
        
        # Intrinsic reward
        rewards = rnd.intrinsic_reward(obs)
        print("RND Intrinsic Reward Shape:", rewards.shape)
        print("RND Intrinsic Reward Values:", rewards)
        
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
             print("RND Output contains NaN or Inf!")
        else:
             print("RND Intrinsic Reward Values OK (Finite)")
             
    except Exception as e:
        print(f"RND Check Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_env_and_policies()
    check_bc_generator()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    check_rnd(device)
