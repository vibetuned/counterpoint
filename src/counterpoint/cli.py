import time
import typer
import gymnasium as gym
import counterpoint.envs # Register envs

app = typer.Typer()

@app.command()
def verify_env(render: bool = False):
    """
    Verify the Piano-v0 environment.
    """
    print("Verifying Piano-v0 environment...")
    from gymnasium.utils.env_checker import check_env
    
    env = gym.make("Piano-v0", render_mode="human" if render else None)
    
    # Check env (Gymnasium utility)
    # Note: check_env might warn about non-standard spaces if strict checks are on, but our Dict is standard.
    print("Running check_env...")
    check_env(env)
    print("check_env passed!")

    # Test Loop
    obs, info = env.reset()
    print("Reset successful. Initial observation keys:", obs.keys())
    
    if render:
        print("Rendering... (Control+C to exit)")
        try:
            for _ in range(5000):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                time.sleep(0.16)
                if terminated or truncated:
                    env.reset()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Render error: {e}")
        finally:
            env.close()

@app.command()
def train():
    """
    Stub for training command.
    """
    print("Training not implemented yet.")

if __name__ == "__main__":
    app()
