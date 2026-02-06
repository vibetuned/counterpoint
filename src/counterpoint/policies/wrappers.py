import gymnasium as gym
import numpy as np

class FlattenActionWrapper(gym.ActionWrapper):
    """
    Flatten Dict action space to MultiDiscrete for SKRL compatibility.
    """
    def __init__(self, env):
        super().__init__(env)
        # 5x Fingers(2) + 5x Black(2) = 10 binary choices
        # No hand_position - it's derived automatically in the env
        nvec = [2]*5 + [2]*5
        self.action_space = gym.spaces.MultiDiscrete(nvec)
        
    def action(self, action):
        """
        Convert the flattened action (array of 10 ints) back to Dict for the environment.
        """
        # Ensure action is numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action)
            
        return {
            "fingers": action[:5],
            "fingers_black": action[5:]
        }
        
    def reverse_action(self, action):
        """
        Convert Dict action to flattened array (if needed).
        """
        return np.concatenate([action["fingers"], action["fingers_black"]])
