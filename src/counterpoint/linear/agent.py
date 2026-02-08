
import numpy as np
import heapq
#from counterpoint.linear.rules import calculate_transition_cost
from counterpoint.rules import calculate_parncutt_cost 
class LinearAgent:
    def __init__(self):
        # Finger indices: 1 to 5
        self.fingers = [1, 2, 3, 4, 5]

    def visualize_step(self, env, save_path):
        """
        Generates and saves a visualization of the graph for the current step.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # 1. Get Data
        score_targets = self._get_lookahead_targets(env)
        if not score_targets:
            return
            
        # 2. Get Anchor
        start_finger, start_note, start_is_black = self._get_anchor_state(env)
        
        # 3. Build Graph
        G = nx.DiGraph()
        
        # Add positions for layout: x=step, y=finger
        pos = {}
        
        # Helper to add node
        def add_node_to_graph(step, finger):
            node_id = f"{step}_{finger}"
            G.add_node(node_id, step=step, finger=finger)
            pos[node_id] = (step, finger)
            return node_id

        # Queue for BFT/DFT to build graph
        # State: (step, finger)
        queue = []
        
        # Init Roots
        if start_finger is None:
            # Step 0 roots
            for f in self.fingers:
                nid = add_node_to_graph(0, f)
                queue.append((0, f))
        else:
            # Root at -1
            root_id = add_node_to_graph(-1, start_finger)
            queue.append((-1, start_finger))
            
        visited = set()
        
        # Build edges forward
        target_len = len(score_targets)
        
        # Just explore everything reachable to allow full visualization
        while queue:
            step, finger = queue.pop(0)
            
            if (step, finger) in visited:
                continue
            visited.add((step, finger))
            
            # If at end, stop
            if step >= target_len - 1:
                continue
                
            next_step = step + 1
            
            # Determine Current and Next Note info for Cost Calc
            if step == -1:
                curr_note = start_note
                curr_is_black = start_is_black
            else:
                curr_note, curr_is_black = score_targets[step]
                
            next_note, next_is_black = score_targets[next_step]
            
            # Constraints
            if curr_note == next_note:
                 next_fingers = [finger]
            else:
                 next_fingers = [f for f in self.fingers if f != finger]
                 
            for next_f in next_fingers:
                cost = calculate_parncutt_cost(
                    prev_finger=finger,
                    prev_note=curr_note,
                    prev_is_black=curr_is_black,
                    curr_finger=next_f,
                    curr_note=next_note,
                    curr_is_black=next_is_black
                )
                
                u = f"{step}_{finger}"
                v = f"{next_step}_{next_f}"
                
                if v not in pos:
                    add_node_to_graph(next_step, next_f)
                    queue.append((next_step, next_f))
                    
                G.add_edge(u, v, weight=cost)

        # 4. Plot
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(G, pos)
        
        # Draw edges with weight labels
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        # Filter labels? Might be too crowded. Round them.
        edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Graph at Step {env.unwrapped._current_step} (Anchor: F{start_finger})")
        plt.xlabel("Lookahead Step")
        plt.ylabel("Finger")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(save_path)
        plt.close()

    def _get_anchor_state(self, env):
        """Helper to extract anchor state."""
        base_env = env.unwrapped
        prev_action = base_env._last_action
        
        prev_finger = None
        if prev_action is not None and "fingers" in prev_action:
            fingers = prev_action["fingers"]
            indices = np.where(fingers == 1)[0]
            if len(indices) > 0:
                prev_finger = indices[0] + 1
        
        prev_note = None
        prev_is_black = False
        if base_env._current_step > 0:
            hist_idx = base_env._current_step - 1
            if hist_idx < len(base_env._score_targets):
                prev_note, prev_is_black = base_env._score_targets[hist_idx]
                
        return prev_finger, prev_note, prev_is_black

    def solve(self, env):
        """
        Solves the best fingering for the current lookahead in the environment.
        Returns the action for the IMMEDIATE next step.
        """
        # 1. Get Lookahead Data
        score_targets = self._get_lookahead_targets(env)
        if not score_targets:
             return self._no_op_action()
             
        # Get Anchor
        start_finger, start_note, start_is_black = self._get_anchor_state(env)

        # 2. Build & Solve Graph (Dijkstra)
        # Returns list of fingerings for the lookahead window: [f_t0, f_t1, ...]
        best_fingering_path = self._dijkstra(score_targets, start_finger, start_note, start_is_black)
        
        # 3. Convert first step of path to Env Action
        if not best_fingering_path:
             return self._no_op_action()
             
        next_finger = best_fingering_path[0]
        # Current target note property
        target_note, is_black = score_targets[0]
        
        return self._create_action(next_finger, is_black)


    def _get_lookahead_targets(self, env):
        """
        Extracts (note, is_black) from env's score_targets for the lookahead window.
        """
        base_env = env.unwrapped
        start_step = base_env._current_step
        max_len = len(base_env._score_targets)
        
        horizon = 10 # Matches env.LOOKAHEAD
        targets = []
        for t in range(horizon):
            idx = start_step + t
            if idx < max_len:
                targets.append(base_env._score_targets[idx])
            else:
                break
        return targets

    def _dijkstra(self, targets, start_finger, start_note, start_is_black):
        """
        Finds shortest path of fingers from 'start_finger' (at 'start_note') 
        through the sequence of 'targets'.
        
        State: (step_idx, finger_idx)
        step_idx: -1 (Root/History) to len(targets)-1
        """
        if not targets:
            return []

        # Priority Queue: (cost, step_idx, finger_idx, path_list)
        pq = []
        
        # Initialization
        if start_finger is None:
            # First note of episode: No history.
            # We treat the first note (step=0) as the "roots".
            # Can start with any of the 5 fingers.
            for f in self.fingers:
                heapq.heappush(pq, (0, 0, f, [f]))
        else:
            # History exists. Root is at step = -1.
            # We expand from Root (-1) to step 0.
            # Logic:
            # Root: (cost=0, step=-1, finger=start_finger, path=[])
            # But the 'path' we return needs to start at step 0.
            
            # Optimization: Pre-calculate the first expansion to avoid managing step=-1 in the main loop
            # and to keep path consistent.
            
            root_finger = start_finger
            root_note = start_note
            root_is_black = start_is_black
            
            curr_note, curr_is_black = targets[0]
            
            if root_note == curr_note:
                 # Same note -> Must use same finger
                 candidates = [root_finger]
            else:
                 # Note Change -> Must use DIFFERENT fingers
                 candidates = [f for f in self.fingers if f != root_finger]
            
            for f in candidates:
                cost = calculate_parncutt_cost(
                    prev_finger=root_finger,
                    prev_note=root_note,
                    prev_is_black=root_is_black,
                    curr_finger=f,
                    curr_note=curr_note,
                    curr_is_black=curr_is_black
                )
                heapq.heappush(pq, (cost, 0, f, [f]))

        visited = {} # (step_idx, finger_idx) -> min_cost
        
        best_final_path = None
        target_len = len(targets)

        while pq:
            cost, step, finger, path = heapq.heappop(pq)
            
            # Global goal check
            if step == target_len - 1:
                return path

            # Pruning
            if (step, finger) in visited and visited[(step, finger)] <= cost:
                continue
            visited[(step, finger)] = cost
            
            # Expansions: Step -> Step + 1
            next_step = step + 1
            if next_step >= target_len:
                continue
                
            curr_note, curr_is_black = targets[step]
            next_note, next_is_black = targets[next_step]
            
            # Constraint: "if the note is the same play the same finger"
            if curr_note == next_note:
                 next_fingers_to_consider = [finger]
            else:
                 # "if changing note we will add 4 node" -> strictly different fingers
                 next_fingers_to_consider = [f for f in self.fingers if f != finger]

            for next_f in next_fingers_to_consider:
                edge_cost = calculate_parncutt_cost(
                    prev_finger=finger, 
                    prev_note=curr_note, 
                    prev_is_black=curr_is_black,
                    curr_finger=next_f, 
                    curr_note=next_note, 
                    curr_is_black=next_is_black
                )
                
                new_cost = cost + edge_cost
                new_path = path + [next_f]
                heapq.heappush(pq, (new_cost, next_step, next_f, new_path))
                
        return best_final_path if best_final_path else []

    def _create_action(self, finger_idx, is_black):
        fingers = np.zeros(5, dtype=np.int8)
        fingers_black = np.zeros(5, dtype=np.int8)
        
        f_i = finger_idx - 1
        fingers[f_i] = 1
        
        # Apply mask logic: if target is black, the active finger MUST be black.
        # If target is white, it MUST be white (0).
        if is_black:
            fingers_black[f_i] = 1
        else:
            fingers_black[f_i] = 0
            
        return {
            "fingers": fingers,
            "fingers_black": fingers_black
        }

    def _no_op_action(self):
        return {
            "fingers": np.zeros(5, dtype=np.int8),
            "fingers_black": np.zeros(5, dtype=np.int8)
        }
