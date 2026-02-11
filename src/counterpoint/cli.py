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
    score_type: str = typer.Option(None, help="Score generator type: scale, chord, arpeggio, mei"),
    mei_path: str = typer.Option(None, help="Path to MEI file(s) when using mei generator"),
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
        rnd_coef=rnd_coef,
        score_type=score_type,
        mei_path=mei_path,
    )

@app.command()
def test(
    checkpoint: str = typer.Option(None, help="Path to checkpoint file"),
    score_type: str = typer.Option(None, help="Score generator type: scale, chord, arpeggio, mei"),
    mei_path: str = typer.Option(None, help="Path to MEI file(s) when using mei generator"),
):
    """
    Test the trained model with rendering.
    """
    from counterpoint.train import test as test_loop
    test_loop(path=checkpoint, score_type=score_type, mei_path=mei_path)

@app.command()
def linear(
    debug: bool = typer.Option(False, help="Enable debug visualization (save graphs)"),
    score_type: str = typer.Option(None, help="Score generator type: scale, chord, arpeggio, mei"),
    mei_path: str = typer.Option(None, help="Path to MEI file(s) when using mei generator"),
    hand: int = typer.Option(None, help="Hand to use (1=RH, 2=LH). Overrides config."),
):
    """
    Run the Linear Debugging Agent (Dijkstra) on the Piano environment.
    """
    from counterpoint.linear import run_linear_agent
    run_linear_agent(debug=debug, score_type=score_type, mei_path=mei_path, hand=hand)

@app.command()
def annotate(
    input: str = typer.Option(..., help="Path to MEI file or directory of MEI files"),
    output: str = typer.Option(..., help="Output directory for annotated MEI files"),
    model: str = typer.Option(None, help="Path to PPO checkpoint (uses trained model)"),
    linear: bool = typer.Option(False, help="Use linear (Dijkstra) agent instead of PPO"),
    rules: str = typer.Option("jacobs", help="Rule set for linear agent: jacobs, parncutt"),
    staff: int = typer.Option(1, help="Staff to annotate (1=treble/RH, 2=bass/LH)"),
    both: bool = typer.Option(False, help="Annotate both staves (RH + LH)"),
):
    """
    Annotate MEI files with piano fingering.
    
    Uses either a trained PPO model (--model) or the linear Dijkstra agent (--linear)
    to compute fingerings and write them into the MEI files.
    
    Examples:
        counterpoint annotate --input score.mei --output ./annotated --linear
        counterpoint annotate --input score.mei --output ./annotated --linear --both
        counterpoint annotate --input ./scores/ --output ./annotated --model checkpoint.pt
    """
    if not model and not linear:
        print("Error: Must specify either --model <checkpoint> or --linear")
        raise typer.Exit(1)
    
    if model and linear:
        print("Error: Cannot use both --model and --linear. Choose one.")
        raise typer.Exit(1)
    
    from counterpoint.annotate import annotate_mei_with_linear, annotate_mei_with_ppo
    
    if linear:
        print(f"Annotating with linear agent (rules={rules})...")
        annotate_mei_with_linear(
            input_path=input,
            output_dir=output,
            rules=rules,
            staff=staff,
            both=both,
        )
    else:
        print(f"Annotating with PPO model: {model}")
        annotate_mei_with_ppo(
            input_path=input,
            output_dir=output,
            checkpoint_path=model,
            staff=staff,
            both=both,
        )
    
    print("\nAnnotation complete!")

if __name__ == "__main__":
    app()
