import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class PianoRenderer:
    def __init__(self, pitch_range, lookahead, render_mode="human"):
        self.pitch_range = pitch_range
        self.lookahead = lookahead
        self.render_mode = render_mode
        
        self.window = None
        self.fig = None
        self.ax_grid = None
        self.ax_piano = None

    def close(self):
        if self.window and self.fig:
            plt.close(self.fig)
            self.window = None
            self.fig = None
            
    def render(self, env):
        if self.render_mode == "rgb_array":
            return self._render_frame(env)
        elif self.render_mode == "human":
            self._render_frame(env)
            plt.pause(0.01)

    def _render_frame(self, env):
        # Setup plot if needed
        if self.window is None:
            self.fig, (self.ax_grid, self.ax_piano) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            self.window = True
            if self.render_mode == "human":
                plt.ion()
                self.fig.show()
            
        # --- Top: Grid State ---
        self.ax_grid.clear()
        self.ax_grid.set_title(f"Score Lookahead | Episode: {env._episode_count} | Step: {env._current_step}")
        self.ax_grid.set_xlim(0, self.pitch_range)
        self.ax_grid.set_ylim(0, self.lookahead)
        self.ax_grid.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
        # Force integer ticks for every key
        self.ax_grid.xaxis.set_major_locator(ticker.MultipleLocator(1))
        self.ax_grid.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # Hide x labels on grid to reduce clutter
        self.ax_grid.set_xticklabels([])
        
        # Access env state directly for now as per refactor strategy
        # We need obs["grid"]
        # But _get_obs() might be expensive if called again? env has it?
        # Let's call _get_obs() or access env internal if possible.
        # Env usually returns obs in step(). The renderer doesn't receive obs.
        # We'll call env._get_obs() assuming it's idempotent.
        obs = env._get_obs()
        grid = obs["grid"] # Shape (2, 52, 10)
        
        # Plot Notes
        for t in range(self.lookahead):
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
        self.ax_piano.set_xlim(0, self.pitch_range)
        self.ax_piano.set_ylim(0, 1)
        self.ax_piano.set_yticks([])
        self.ax_piano.axis('off')
        
        # Draw Piano Keys
        has_black = [True, True, False, True, True, True, False] # C to B
        
        # Draw all White keys first
        for i in range(self.pitch_range):
            rect = plt.Rectangle((i, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=1)
            self.ax_piano.add_patch(rect)
        
        # Draw Black keys
        for i in range(self.pitch_range):
            pc = i % 7
            if has_black[pc]:
                # Draw black key between i and i+1
                # Center X ~= i + 1.0
                rect = plt.Rectangle((i + 0.65, 0.35), 0.7, 0.65, facecolor='black', edgecolor='black', zorder=10)
                self.ax_piano.add_patch(rect)

        # Draw Hand (Yellow dot on anchor)
        anchor_x = env._hand_pos + 0.5
        anchor_y = 0.0
        
        # Draw Fingers
        # Default: 5 consecutive white keys starting at hand_pos
        current_fingers = env._last_action["fingers"] if hasattr(env, "_last_action") and env._last_action is not None else [0]*5
        current_blacks = env._last_action["fingers_black"] if hasattr(env, "_last_action") and env._last_action is not None else [0]*5
        
        for i in range(5):
            finger_num = i + 1
            key_idx = env._hand_pos + i
            if key_idx >= self.pitch_range:
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
