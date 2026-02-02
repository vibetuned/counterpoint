
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

from counterpoint.envs.render.piano_render import PianoRenderer
from counterpoint.envs.rewards import RewardMixing, MovementPenalty, BlackKeyChangePenalty, AccuracyReward

class PianoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # --- Constants ---
        self.PITCH_RANGE = 52 # Number of white keys (columns)
        self.ROWS = 2 # 0: Natural, 1: Accidental
        self.LOOKAHEAD = 10
        self.render_mode = render_mode
        
        # --- Action Space ---
        self.action_space = spaces.Dict({
            "hand_position": spaces.Discrete(self.PITCH_RANGE),
            "fingers": spaces.MultiBinary(5),
            "fingers_black": spaces.MultiBinary(5)
        })

        # --- Observation Space ---
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.ROWS, self.PITCH_RANGE, self.LOOKAHEAD), dtype=np.float32),
            "hand_state": spaces.Box(low=0, high=self.PITCH_RANGE, shape=(1,), dtype=np.float32)
        })

        # State
        self._current_step = 0
        self._hand_pos = 0 # Current anchor
        self._episode_count = -1 
        self._last_action = None
        
        # Helper Modules
        self.renderer = PianoRenderer(self.PITCH_RANGE, self.LOOKAHEAD, self.render_mode)
        self.reward_function = RewardMixing()
        self.reward_function.add(MovementPenalty())
        self.reward_function.add(BlackKeyChangePenalty())
        self.reward_function.add(AccuracyReward())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        
        self._current_step = 0
        self._hand_pos = self.np_random.integers(0, self.PITCH_RANGE - 5) 
        self._last_action = None
        
        self._generate_simple_score()
        
        return self._get_obs(), {}

    def _generate_simple_score(self):
        # 1. Determine Length (5 to 12)
        length = self.np_random.integers(5, 13)
        start_note = self.np_random.integers(10, self.PITCH_RANGE - 15)
        direction = 1
        current_note = start_note
        
        self._score_targets = []
        if length > 4:
            reverse_at = self.np_random.integers(2, length - 2)
        else:
            reverse_at = -1

        for i in range(length):
            # Record Target: (Note Index, Is Black?)
            self._score_targets.append((current_note, 0)) # 0 for Natural
            
            if i == reverse_at:
                direction *= -1
            current_note += direction

    def step(self, action):
        target_hand_pos = action["hand_position"]
        
        # Logic delegated to RewardMixing
        # Note: We pass raw action. Reward calc uses self._hand_pos (current) and self._last_action (prev)
        # But we must NOT update self._hand_pos until AFTER reward calc 
        # (because Penalty uses distance from current to target)
        
        total_reward, success = self.reward_function.calculate(self, action)
        
        terminated = False
        truncated = False
        
        if self._current_step < len(self._score_targets):
             if success:
                 # Logic for success (correct note played)
                 self._current_step += 1
                 if self._current_step >= len(self._score_targets):
                      terminated = True
                      total_reward += 50.0
             else:
                 # Logic for failure (wrong note or invalid)
                 # In this env, !success means failure if step < len
                 terminated = True
        else:
            terminated = True 

        # Update State
        self._hand_pos = target_hand_pos
        self._last_action = action
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), total_reward, terminated, truncated, {}

    def _get_obs(self):
        # Construct Grid from Targets
        grid = np.zeros((self.ROWS, self.PITCH_RANGE, self.LOOKAHEAD), dtype=np.float32)
        
        for t in range(self.LOOKAHEAD):
            idx = self._current_step + t
            if idx < len(self._score_targets):
                note, is_black = self._score_targets[idx]
                
                # Determine Row (Natural/Accidental)
                if 0 <= note < self.PITCH_RANGE:
                    if is_black:
                        grid[1, note, t] = 1.0 # Accidental Row
                    else:
                        grid[0, note, t] = 1.0 # Natural Row
                    
        return {
            "grid": grid,
            "hand_state": np.array([self._hand_pos], dtype=np.float32)
        }

    def close(self):
        self.renderer.close()

    def render(self):
        return self.renderer.render(self)
