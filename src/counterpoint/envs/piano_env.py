
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

class PianoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # --- Constants ---
        self.PITCH_RANGE = 52 # Number of white keys (columns)
        self.ROWS = 2 # 0: Natural, 1: Accidental
        self.LOOKAHEAD = 10
        self.render_mode = render_mode
        self.window = None # Plots

        # --- Action Space ---
        # hand_position: The 'anchor' index on the white keys (0-51)
        # fingers: 5 binary flags (1=pressed, 0=released)
        # fingers_black: 5 binary flags (1=black key, 0=white key)
        self.action_space = spaces.Dict({
            "hand_position": spaces.Discrete(self.PITCH_RANGE),
            "fingers": spaces.MultiBinary(5),
            "fingers_black": spaces.MultiBinary(5)
        })

        # --- Observation Space ---
        # grid: The score slice (2 rows, 52 cols, n lookahead)
        # hand_state: Current anchor position
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(self.ROWS, self.PITCH_RANGE, self.LOOKAHEAD), dtype=np.float32),
            "hand_state": spaces.Box(low=0, high=self.PITCH_RANGE, shape=(1,), dtype=np.float32)
        })

        # State
        self._current_step = 0
        self._hand_pos = 0 # Current anchor
        self._score = None # Load score here later

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._current_step = 0
        self._hand_pos = self.np_random.integers(0, self.PITCH_RANGE)
        self._last_action = None
        
        # Stub: Generate a random dummy score for now
        # Shape: (Rows, Cols, Total Length)
        # We'll just generate the lookahead slice dynamically in step for this stub
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Parse Action
        target_hand_pos = action["hand_position"]
        fingers_pressed = action["fingers"]
        fingers_black = action["fingers_black"]
        
        # Update State
        self._hand_pos = target_hand_pos
        self._current_step += 1
        self._last_action = action # Store for rendering
        
        # 2. Compute Reward (Stub)
        # In a real scenario, we check if pressed keys match the current score slice at time t=0
        reward = 0.0
        
        # 3. Check Termination
        terminated = False
        truncated = False
        
        obs = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Stub: Random grid for testing
        grid = np.zeros((self.ROWS, self.PITCH_RANGE, self.LOOKAHEAD), dtype=np.float32)
        
        # Add some random notes
        # Generate indices for valid notes
        # row: 0 or 1
        # col: 0 to 51
        # time: 0 to 9
        for t in range(self.LOOKAHEAD):
            if self.np_random.random() > 0.8: # 20% chance of a note
                r = self.np_random.integers(0, 2)
                c = self.np_random.integers(0, self.PITCH_RANGE)
                grid[r, c, t] = 1.0
                
        return {
            "grid": grid,
            "hand_state": np.array([self._hand_pos], dtype=np.float32)
        }

    def close(self):
        if self.window and self.fig:
            plt.close(self.fig)
            self.window = None
            self.fig = None

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            plt.pause(0.01)

    def _render_frame(self):
        import matplotlib.ticker as ticker
        
        # Setup plot if needed
        if self.window is None:
            self.fig, (self.ax_grid, self.ax_piano) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            self.window = True
            if self.render_mode == "human":
                plt.ion()
                self.fig.show()
            
        # --- Top: Grid State ---
        self.ax_grid.clear()
        self.ax_grid.set_title("Score Lookahead")
        self.ax_grid.set_xlim(0, self.PITCH_RANGE)
        self.ax_grid.set_ylim(0, self.LOOKAHEAD)
        self.ax_grid.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
        # Force integer ticks for every key
        self.ax_grid.xaxis.set_major_locator(ticker.MultipleLocator(1))
        self.ax_grid.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # Hide x labels on grid to reduce clutter
        self.ax_grid.set_xticklabels([])
        
        obs = self._get_obs()
        grid = obs["grid"] # Shape (2, 52, 10)
        
        # Plot Notes
        for t in range(self.LOOKAHEAD):
            # Check Natural (Row 0)
            naturals = np.where(grid[0, :, t] > 0)[0]
            for n in naturals:
                self.ax_grid.broken_barh([(n, 1)], (t, 0.8), facecolors='blue')
                
            # Check Accidental (Row 1)
            accidentals = np.where(grid[1, :, t] > 0)[0]
            for a in accidentals:
                 # Visually offset black notes slightly (conceptually) or just draw them
                 # Using the same x-coord logic as piano keys (between i and i+1)
                 # Black key visual center is roughly i + 1.0
                 self.ax_grid.broken_barh([(a + 0.65, 0.7)], (t, 0.8), facecolors='black')

        # --- Bottom: Piano & Hand ---
        self.ax_piano.clear()
        self.ax_piano.set_title("Piano State")
        self.ax_piano.set_xlim(0, self.PITCH_RANGE)
        self.ax_piano.set_ylim(0, 1)
        self.ax_piano.set_yticks([])
        self.ax_piano.axis('off')
        
        # Draw Piano Keys
        has_black = [True, True, False, True, True, True, False] # C to B
        
        # Draw all White keys first
        for i in range(self.PITCH_RANGE):
            rect = plt.Rectangle((i, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=1)
            self.ax_piano.add_patch(rect)
        
        # Draw Black keys
        # Store black key centers for finger mapping
        black_key_centers = {} 
        
        for i in range(self.PITCH_RANGE):
            pc = i % 7
            if has_black[pc]:
                # Draw black key between i and i+1
                # Center X ~= i + 1.0
                rect = plt.Rectangle((i + 0.65, 0.35), 0.7, 0.65, facecolor='black', edgecolor='black', zorder=10)
                self.ax_piano.add_patch(rect)
                black_key_centers[i] = i + 1.0

        # Draw Hand (Yellow dot on anchor)
        anchor_x = self._hand_pos + 0.5
        anchor_y = 0.0
        
        # Draw Fingers
        # Default: 5 consecutive white keys starting at hand_pos
        current_fingers = self._last_action["fingers"] if hasattr(self, "_last_action") and self._last_action is not None else [0]*5
        current_blacks = self._last_action["fingers_black"] if hasattr(self, "_last_action") and self._last_action is not None else [0]*5
        
        for i in range(5):
            finger_num = i + 1
            key_idx = self._hand_pos + i
            if key_idx >= self.PITCH_RANGE:
                continue

            # Determine coordinates
            is_black = current_blacks[i]
            is_pressed = current_fingers[i]
            
            # X Coordinate
            if is_black and (key_idx % 7 in [0, 1, 3, 4, 5]): # Check if black key exists here
                 # Use black key center
                 # Simple heuristic: i + 1.0 or use the map
                 x = key_idx + 1.0 
                 y = 0.5 # Higher up on the key
                 color_bg = 'black' # Contrast
            else:
                 x = key_idx + 0.5
                 y = 0.2 # Slightly higher than anchor (0.2)
                 color_bg = 'white'
            
            # Connect Anchor to Finger
            self.ax_piano.plot([anchor_x, x], [anchor_y, y], color='gold', linewidth=1.5, alpha=0.7, zorder=25)

            # Style based on press state
            # If pressed: Solid Fill
            # If not: Hollow or Light
            
            facecolor = 'red' if is_pressed else 'white'
            edgecolor = 'red'
            
            # Draw Marker
            self.ax_piano.plot([x], [y], marker='o', markersize=12, 
                               markerfacecolor=facecolor, markeredgecolor=edgecolor, 
                               label=f'F{finger_num}', zorder=30)
            
            # Draw Number
            text_color = 'white' if is_pressed else 'red'
            self.ax_piano.text(x, y, str(finger_num), color=text_color, 
                               ha='center', va='center', fontsize=8, fontweight='bold', zorder=31)

        # Plot Anchor on top of lines
        self.ax_piano.plot([anchor_x], [anchor_y], 'yo', markersize=14, markeredgecolor='orange', label='Hand', zorder=35)

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
