
import gymnasium as gym
import time
from counterpoint.linear.agent import LinearAgent

def run_linear_agent(debug=False, score_type=None, mei_path=None, hand=None, rules="jacobs"):
    """
    Run the Linear Debugging Agent (Dijkstra) on the Piano environment.
    Runs continuously over multiple episodes.
    """
    from counterpoint.train import load_config, _env_kwargs_from_config
    
    config = load_config()
    env_kwargs = _env_kwargs_from_config(
        config, 
        score_type_override=score_type,
        mei_path_override=mei_path,
        hand_override=hand,
    )
    
    hand_label = "LH" if env_kwargs.get("hand", 1) == 2 else "RH"
    
    # Initialize Env with config-driven kwargs
    env = gym.make("Piano-v0", render_mode="human", **env_kwargs)
    
    agent = LinearAgent(rules=rules)
    
    print(f"Starting Linear Debugging Agent ({hand_label})...")
    print(f"Generator: {env_kwargs.get('score_generator_type', 'arpeggio')}")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            # Reset
            obs, _ = env.reset()
            terminated = False
            truncated = False
            
            print(f"Episode: {env.unwrapped._episode_count}")
            
            # --- DEBUG VISUALIZATION ---
            if debug:
                # Capture first step of each episode
                import os
                os.makedirs("debug_linear", exist_ok=True)
                episode_num = env.unwrapped._episode_count
                
                # Save Graph Plan
                agent.visualize_step(env, f"debug_linear/ep_{episode_num}_graph.png")
                
                # Save Render of initial state
                # Force a render update without stepping
                env.render() # Updates the figure
                if hasattr(env.unwrapped.renderer, 'fig') and env.unwrapped.renderer.fig:
                     env.unwrapped.renderer.fig.savefig(f"debug_linear/ep_{episode_num}_env.png")
            
            while not (terminated or truncated):
                # Solve for next action
                action = agent.solve(env)
                
                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # PianoEnv step() calls render() if render_mode is human.
                
                time.sleep(0.2) # Slow down to watch
                
            print(f"Episode finished. Total Reward: {getattr(env.unwrapped, '_episode_reward', 0.0):.2f}")
            time.sleep(0.2) # Pause between episodes
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
