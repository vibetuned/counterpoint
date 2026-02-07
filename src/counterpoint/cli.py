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
def train(
    steps: int = typer.Option(100000, help="Number of training steps"),
    resume: str = typer.Option(None, help="Path to checkpoint file to resume from"),
    use_bc: bool = typer.Option(None, help="Enable Behavior Cloning (overrides config)"),
    use_rnd: bool = typer.Option(None, help="Enable RND exploration (overrides config)"),
    bc_coef: float = typer.Option(None, help="BC loss coefficient (overrides config)"),
    rnd_coef: float = typer.Option(None, help="RND reward coefficient (overrides config)"),
):
    """
    Run SKRL PPO training with optional BC and RND exploration.
    """
    from counterpoint.train import train as train_loop
    train_loop(
        steps=steps, 
        resume_path=resume,
        use_bc=use_bc,
        use_rnd=use_rnd,
        bc_coef=bc_coef,
        rnd_coef=rnd_coef
    )

@app.command()
def test(checkpoint: str = typer.Option(None, help="Path to checkpoint file")):
    """
    Test the trained model with rendering.
    """
    from counterpoint.train import test as test_loop
    test_loop(path=checkpoint)

@app.command()
def linear(debug: bool = typer.Option(False, help="Enable debug visualization (save graphs)")):
    """
    Run the Linear Debugging Agent (Dijkstra) on the Piano environment.
    """
    from counterpoint.linear import run_linear_agent
    run_linear_agent(debug=debug)

if __name__ == "__main__":
    app()
