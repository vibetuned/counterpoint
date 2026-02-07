
import numpy as np
from typing import List, Tuple

from .scores import SCALE_FINGERINGS, MAJOR_SCALES, SCALE_ROOTS

class DemonstrationGenerator:
    """
    Generates expert demonstrations for behavior cloning.
    
    Uses classical fingering patterns to create (observation, expert_action) pairs
    that can be used to train the policy via behavior cloning loss.
    """
    
    def __init__(self, pitch_range: int = 52, lookahead: int = 10):
        self.pitch_range = pitch_range
        self.lookahead = lookahead
        self.scale_names = list(SCALE_FINGERINGS.keys())
        self.rng = np.random.default_rng()
    
    def _build_observation(self, score_targets: List[Tuple[int, int]], 
                           step: int, hand_pos: int, prev_finger: int = None, prev_note: int = None) -> dict:
        """Build observation dict matching PianoEnv format.
        
        Args:
            score_targets: List of (note, is_black) tuples
            step: Current step in the score
            hand_pos: Current hand position
            prev_finger: Previous finger used (1-5, None if first step)
            prev_note: Previous note played (None if first step)
        """
        grid = np.zeros((2, self.pitch_range, self.lookahead), dtype=np.float32)
        
        for t in range(self.lookahead):
            idx = step + t
            if idx < len(score_targets):
                note, is_black = score_targets[idx]
                if 0 <= note < self.pitch_range:
                    row = 1 if is_black else 0
                    grid[row, note, t] = 1.0
        
        # Compute relative target
        if step < len(score_targets):
            target_note, is_black = score_targets[step]
            relative_target = float(target_note - hand_pos)
            
            # Action mask generation (mimics PianoEnv)
            # num_notes calculation (count of notes at t=0 in grid)
            num_notes = 1.0 # Scale demos are single note
            
            # fingers_black_mask
            if is_black:
                fingers_black_mask = np.ones(5, dtype=np.float32)
            else:
                fingers_black_mask = np.zeros(5, dtype=np.float32)
            
            # finger_mask: forbid previous finger if note changed
            finger_mask = np.ones(5, dtype=np.float32)  # Default: all allowed
            if prev_finger is not None and prev_note is not None:
                if prev_note != target_note:
                    finger_mask[prev_finger - 1] = 0.0  # Forbid previous finger
                
        else:
            relative_target = 0.0
            num_notes = 1.0
            fingers_black_mask = np.zeros(5, dtype=np.float32)
            finger_mask = np.ones(5, dtype=np.float32)
        
        # Concatenate: [fingers_black_mask(5), num_notes(1), finger_mask(5)] = 11 values
        action_mask = np.concatenate([
            fingers_black_mask,
            np.array([num_notes], dtype=np.float32),
            finger_mask
        ])
        
        return {
            "grid": grid,
            "hand_state": np.array([hand_pos], dtype=np.float32),
            "relative_target": np.array([relative_target], dtype=np.float32),
            "action_mask": action_mask
        }
    
    def _finger_to_action(self, finger: int, is_black: bool) -> np.ndarray:
        """
        Convert finger number (1-5) to action array.
        
        Returns: array of shape (10,) = [5 fingers, 5 black modifiers]
        """
        action = np.zeros(10, dtype=np.int32)
        finger_idx = finger - 1  # Convert 1-indexed to 0-indexed
        action[finger_idx] = 1  # Press this finger
        if is_black:
            action[5 + finger_idx] = 1  # Black key modifier
        return action
    
    def generate_demo(self, scale_name: str = None, hand: str = "RH", 
                      octave: int = None) -> List[Tuple[dict, np.ndarray]]:
        """
        Generate a full demonstration for a scale.
        
        Args:
            scale_name: Scale to demonstrate (random if None)
            hand: "RH" (right hand) or "LH" (left hand)
            octave: Starting octave (random if None)
            
        Returns:
            List of (observation, expert_action) tuples
        """
        if scale_name is None:
            scale_name = self.rng.choice(self.scale_names)
        
        if scale_name not in SCALE_FINGERINGS:
            raise ValueError(f"Unknown scale: {scale_name}")
        
        fingering = SCALE_FINGERINGS[scale_name][hand]
        scale_pattern = MAJOR_SCALES[scale_name]
        root_offset = SCALE_ROOTS[scale_name]
        
        # Determine starting octave
        if octave is None:
            max_octave = (self.pitch_range - 14) // 7
            octave = self.rng.integers(1, max(2, max_octave))
        
        base_column = octave * 7 + root_offset
        
        # Build scale notes for 2 octaves (matching fingering length)
        score_targets = []
        for i, finger in enumerate(fingering):
            scale_idx = i % 7
            octave_offset = (i // 7) * 7
            col_offset, is_black = scale_pattern[scale_idx]
            column = base_column + col_offset + octave_offset
            
            if 0 <= column < self.pitch_range:
                score_targets.append((column, is_black))
        
        # Generate demonstrations
        demos = []
        hand_pos = base_column  # Start hand at root
        prev_finger = None
        prev_note = None
        
        for step, (note, is_black) in enumerate(score_targets):
            if step >= len(fingering):
                break
                
            finger = fingering[step]
            
            # Build observation with previous finger info for finger_mask
            obs = self._build_observation(score_targets, step, hand_pos, 
                                          prev_finger=prev_finger, prev_note=prev_note)
            
            # Build expert action
            action = self._finger_to_action(finger, is_black)
            
            demos.append((obs, action))
            
            # Update hand position (leftmost finger aligns with note)
            hand_pos = note - (finger - 1)
            
            # Track for next iteration
            prev_finger = finger
            prev_note = note
        
        return demos
    
    def sample_batch(self, batch_size: int = 64) -> Tuple[dict, np.ndarray]:
        """
        Sample a batch of (observation, action) pairs for BC training.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            (observations_dict, actions_array) where observations_dict has
            batched arrays and actions_array is shape (batch, 10)
        """
        # Collect samples
        all_obs = {"grid": [], "hand_state": [], "relative_target": [], "action_mask": []}
        all_actions = []
        
        while len(all_actions) < batch_size:
            # Generate a random demo
            scale = self.rng.choice(self.scale_names)
            hand = "RH" #self.rng.choice(["RH", "LH"])
            demos = self.generate_demo(scale_name=scale, hand=hand)
            
            for obs, action in demos:
                all_obs["grid"].append(obs["grid"])
                all_obs["hand_state"].append(obs["hand_state"])
                all_obs["relative_target"].append(obs["relative_target"])
                all_obs["action_mask"].append(obs["action_mask"])
                all_actions.append(action)
                
                if len(all_actions) >= batch_size:
                    break
        
        # Stack into batched arrays
        batched_obs = {
            "grid": np.stack(all_obs["grid"][:batch_size]),
            "hand_state": np.stack(all_obs["hand_state"][:batch_size]),
            "relative_target": np.stack(all_obs["relative_target"][:batch_size]),
            "action_mask": np.stack(all_obs["action_mask"][:batch_size])
        }
        batched_actions = np.stack(all_actions[:batch_size])
        
        return batched_obs, batched_actions
    
    def flatten_observation(self, obs: dict) -> np.ndarray:
        """Flatten observation dict to match SKRL format (alphabetical: action_mask, grid, hand, relative)."""
        grid_flat = obs["grid"].flatten()
        
        # Order: [action_mask(6), grid(1040), hand_state(1), relative_target(1)]
        return np.concatenate([
            obs["action_mask"],
            grid_flat, 
            obs["hand_state"], 
            obs["relative_target"]
        ])
    
    def sample_batch_flat(self, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample batch with flattened observations (for SKRL compatibility).
        
        Returns:
            (observations, actions) where observations is shape (batch, 1042)
            and actions is shape (batch, 10)
        """
        obs_dict, actions = self.sample_batch(batch_size)
        
        # Flatten each observation
        batch_obs = []
        for i in range(batch_size):
            obs = {
                "grid": obs_dict["grid"][i],
                "hand_state": obs_dict["hand_state"][i],
                "relative_target": obs_dict["relative_target"][i],
                "action_mask": obs_dict["action_mask"][i]
            }
            batch_obs.append(self.flatten_observation(obs))
        
        return np.stack(batch_obs), actions
