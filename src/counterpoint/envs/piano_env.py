
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

from counterpoint.envs.render.piano_render import PianoRenderer
from counterpoint.envs.rewards import (
    RewardMixing, MovementPenalty, WrongColorPenalty, AccuracyReward, 
    CompletionReward, KeyChangePenalty, FingerRepetitionPenalty,
    ArpeggioReward, NoteProgressReward
)
from counterpoint.data.scores import MajorScaleGenerator


class PianoEnv(gym.Env):
    """
    Piano fingering environment.
    
    The agent learns fingering patterns only - hand position is automatically
    derived from the leftmost pressed finger aligning with the leftmost target note.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # --- Constants ---
        self.PITCH_RANGE = 52  # Number of white keys (columns)
        self.ROWS = 2  # 0: Natural, 1: Accidental
        self.LOOKAHEAD = 10
        self.render_mode = render_mode
        
        # --- Action Space (fingering only, no hand position) ---
        self.action_space = spaces.Dict({
            "fingers": spaces.MultiBinary(5),        # Which fingers to press
            "fingers_black": spaces.MultiBinary(5)   # Black key modifier per finger
        })

        # --- Observation Space ---
        # action_mask: 6 values - [fingers_black(5), num_notes(1)]
        # fingers_black: 0 = white key, 1 = black key (for masking key color)
        # num_notes: number of notes at current timestep (for priority head)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.ROWS, self.PITCH_RANGE, self.LOOKAHEAD), dtype=np.float32),
            "hand_state": spaces.Box(low=0, high=self.PITCH_RANGE, shape=(1,), dtype=np.float32),
            "relative_target": spaces.Box(low=-self.PITCH_RANGE, high=self.PITCH_RANGE, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=5, shape=(6,), dtype=np.float32)
        })

        # State
        self._current_step = 0  # Current position in score (note index)
        self._step_count = 0     # Total actions taken in episode
        self._hand_pos = 0  # Current hand anchor position
        self._episode_count = -1 
        self._last_action = None
        self._derived_hand_pos = 0  # Hand position derived from current action
        self._max_steps_multiplier = 3  # Max steps = score_length * this
        
        # Helper Modules
        self.renderer = PianoRenderer(self.PITCH_RANGE, self.LOOKAHEAD, self.render_mode)
        self.score_generator = MajorScaleGenerator(self.PITCH_RANGE)
        self.reward_function = RewardMixing()
        self.reward_function.add(MovementPenalty())
        self.reward_function.add(KeyChangePenalty())
        self.reward_function.add(WrongColorPenalty())
        self.reward_function.add(FingerRepetitionPenalty())
        self.reward_function.add(ArpeggioReward())  # Encourages finger variety
        self.reward_function.add(AccuracyReward())
        #self.reward_function.add(NoteProgressReward(bonus=1.0))  # Intermediate progress
        #self.reward_function.add(CompletionReward(bonus=50.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        
        self._current_step = 0
        self._step_count = 0
        self._last_action = None
        
        self._score_targets = self.score_generator.generate(self.np_random)
        
        # Initialize hand position based on first target
        if self._score_targets:
            first_note, _ = self._score_targets[0]
            self._hand_pos = first_note  # Start with hand at first note
        else:
            self._hand_pos = self.PITCH_RANGE // 2
        
        self._derived_hand_pos = self._hand_pos
        
        return self._get_obs(), {}

    def _compute_hand_position(self, action):
        """
        Derive hand position from fingering action.
        
        Rule: Leftmost pressed finger aligns with leftmost target note.
        """
        fingers = action["fingers"]
        pressed_indices = [i for i, f in enumerate(fingers) if f == 1]
        
        if not pressed_indices:
            # No fingers pressed - no change
            return self._hand_pos
        
        leftmost_finger = min(pressed_indices)
        
        # Get current target note
        if self._current_step < len(self._score_targets):
            target_note, _ = self._score_targets[self._current_step]
            # Place hand so leftmost finger hits target note
            return target_note - leftmost_finger
        
        return self._hand_pos

    def step(self, action):
        self._step_count += 1  # Track total actions
        
        # Compute derived hand position from fingering
        self._derived_hand_pos = self._compute_hand_position(action)
        
        # Create augmented action with derived hand position for reward calculation
        augmented_action = {
            "hand_position": self._derived_hand_pos,
            "fingers": action["fingers"],
            "fingers_black": action["fingers_black"]
        }
        
        # Calculate rewards using the derived hand position
        total_reward, success = self.reward_function.calculate(self, augmented_action)
        
        terminated = False
        truncated = False
        
        # Check for max steps (truncation to prevent infinite episodes)
        max_steps = len(self._score_targets) * self._max_steps_multiplier
        if self._step_count >= max_steps:
            truncated = True
        
        if self._current_step < len(self._score_targets):
            if success:
                # Correct note played - advance to next
                self._current_step += 1
                if self._current_step >= len(self._score_targets):
                    terminated = True  # Score completed!
            # Wrong note: DON'T terminate, just don't advance
            # Agent is penalized and must try again on same note
            # This maintains ergodicity and allows full exploration
        else:
            terminated = True 

        # Update state with derived hand position
        self._hand_pos = self._derived_hand_pos
        self._last_action = augmented_action  # Store augmented for next step's rewards
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), total_reward, terminated, truncated, {}

    def get_action_mask(self, grid=None):
        """
        Generate action masks based on current target note and grid.
        
        Mask logic:
        - num_notes: count of notes at current timestep (sum of grid[:, :, 0])
        - fingers_black_mask: force key color based on target note
        
        Returns:
            Dict with:
            - fingers_black_mask: 5 values (0 = white, 1 = black)
            - num_notes: number of notes to play (for priority head)
        """
        # Calculate num_notes from grid if provided
        if grid is not None:
            # Sum across both rows and all pitches at current timestep (t=0)
            num_notes = int(grid[:, :, 0].sum())
        else:
            num_notes = 1  # Default to single note
        
        # Ensure at least 1 finger and at most 5
        num_notes = max(1, min(5, num_notes))
        
        if self._current_step < len(self._score_targets):
            _, target_is_black = self._score_targets[self._current_step]
            
            if target_is_black:
                # Target is BLACK: force black keys
                fingers_black_mask = np.ones(5, dtype=np.float32)
            else:
                # Target is WHITE: force white keys
                fingers_black_mask = np.zeros(5, dtype=np.float32)
        else:
            # No target - default to white
            fingers_black_mask = np.zeros(5, dtype=np.float32)
        
        return {
            "fingers_black_mask": fingers_black_mask,
            "num_notes": num_notes
        }

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
                        grid[1, note, t] = 1.0  # Accidental Row
                    else:
                        grid[0, note, t] = 1.0  # Natural Row
        
        # Compute relative target position
        if self._current_step < len(self._score_targets):
            target_note, _ = self._score_targets[self._current_step]
            relative_target = float(target_note - self._hand_pos)
        else:
            relative_target = 0.0
        
        # Get action mask with num_notes calculated from grid
        mask = self.get_action_mask(grid=grid)
        # Concatenate: [fingers_black_mask(5), num_notes(1)] = 6 values
        action_mask = np.concatenate([
            mask["fingers_black_mask"],
            np.array([mask["num_notes"]], dtype=np.float32)
        ])
                    
        return {
            "grid": grid,
            "hand_state": np.array([self._hand_pos], dtype=np.float32),
            "relative_target": np.array([relative_target], dtype=np.float32),
            "action_mask": action_mask
        }

    def close(self):
        self.renderer.close()

    def render(self):
        return self.renderer.render(self)
