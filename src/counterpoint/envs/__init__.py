from gymnasium.envs.registration import register

register(
    id="Piano-v0",
    entry_point="counterpoint.envs.piano_env:PianoEnv",
    max_episode_steps=1000,
    kwargs={},  # All kwargs passed through to PianoEnv.__init__
)
