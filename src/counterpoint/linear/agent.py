
import numpy as np
from counterpoint.rules import calculate_jacobs_cost, calculate_parncutt_cost

_COST_FN_MAP = {
    "jacobs": calculate_jacobs_cost,
    "parncutt": calculate_parncutt_cost,
}

class LinearAgent:
    def __init__(self, rules: str = "jacobs"):
        # Finger indices: 1 to 5
        self.fingers = [1, 2, 3, 4, 5]
        if rules not in _COST_FN_MAP:
            raise ValueError(f"Unknown rules '{rules}'. Choose from: {list(_COST_FN_MAP.keys())}")
        self._cost_fn = _COST_FN_MAP[rules]
        self._rules_name = rules

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
                cost = self._cost_fn(
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
                target = base_env._score_targets[hist_idx]
                # Normalize chords to single note (LH: rightmost, RH: leftmost)
                if isinstance(target[0], tuple):
                    note = target[-1] if base_env.hand == 2 else target[0]
                    prev_note, prev_is_black = note
                else:
                    prev_note, prev_is_black = target
                
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
        Normalizes chords to their first (leftmost) note to match env behavior.
        """
        base_env = env.unwrapped
        start_step = base_env._current_step
        max_len = len(base_env._score_targets)
        
        horizon = 10 # Matches env.LOOKAHEAD
        targets = []
        for t in range(horizon):
            idx = start_step + t
            if idx < max_len:
                target = base_env._score_targets[idx]
                # Normalize: chords are tuple-of-tuples, single notes are (int, int)
                if isinstance(target[0], tuple):
                    # LH: use rightmost note, RH: use leftmost note
                    targets.append(target[-1] if base_env.hand == 2 else target[0])
                else:
                    targets.append(target)
            else:
                break
        return targets

    def _dijkstra(self, targets, start_finger, start_note, start_is_black):
        """
        Finds shortest path of fingers through the note sequence using an
        'exploded network' where each node represents a (step, prev_finger,
        curr_finger) triple.  This allows every edge to evaluate the FULL
        cost function with 3-note context (prev_prev, prev, curr, next).

        Uses NetworkX for graph construction and shortest-path search.
        """
        import networkx as nx

        if not targets:
            return []

        target_len = len(targets)
        G = nx.DiGraph()

        SOURCE = "source"
        SINK = "sink"
        G.add_node(SOURCE)
        G.add_node(SINK)

        # -----------------------------------------------------------------
        # STEP 0: Connect source to initial nodes
        # -----------------------------------------------------------------
        first_note, first_is_black = targets[0]

        if start_finger is None:
            # No history — any finger can start, no prev context
            for f in self.fingers:
                node = (0, None, f)
                G.add_node(node)
                G.add_edge(SOURCE, node, weight=0)
        else:
            # History exists — expand from anchor to step 0
            if start_note == first_note:
                candidates = [start_finger]
            else:
                candidates = [f for f in self.fingers if f != start_finger]

            for f in candidates:
                cost = self._cost_fn(
                    prev_finger=start_finger,
                    prev_note=start_note,
                    prev_is_black=start_is_black,
                    curr_finger=f,
                    curr_note=first_note,
                    curr_is_black=first_is_black,
                    hand=getattr(self, '_hand', 1),
                )
                node = (0, start_finger, f)
                G.add_node(node)
                G.add_edge(SOURCE, node, weight=cost)

        # -----------------------------------------------------------------
        # STEPS 1 .. target_len-1: Build exploded edges
        # -----------------------------------------------------------------
        for step in range(1, target_len):
            prev_note, prev_is_black = targets[step - 1]
            curr_note, curr_is_black = targets[step]

            # Next-note context (for Rule 10, etc.)
            if step + 1 < target_len:
                next_note_val, next_is_black_val = targets[step + 1]
            else:
                next_note_val, next_is_black_val = None, None

            # Finger constraint for this step
            if prev_note == curr_note:
                # Same note → same finger
                allowed_fn = lambda prev_f: [prev_f]
            else:
                # Different note → different finger
                allowed_fn = lambda prev_f: [f for f in self.fingers if f != prev_f]

            # Iterate over all nodes at the previous step
            prev_step_nodes = [n for n in G.nodes if isinstance(n, tuple) and len(n) == 3 and n[0] == step - 1]

            for prev_node in prev_step_nodes:
                _, pp_finger, prev_finger = prev_node  # pp = prev_prev

                for curr_finger in allowed_fn(prev_finger):
                    curr_node = (step, prev_finger, curr_finger)
                    if curr_node not in G:
                        G.add_node(curr_node)

                    # Determine prev_prev note for Rule 4/5 context
                    if step >= 2:
                        pp_note, pp_is_black = targets[step - 2]
                    elif start_finger is not None:
                        pp_note, pp_is_black = start_note, start_is_black
                    else:
                        pp_note, pp_is_black = None, None

                    cost = self._cost_fn(
                        prev_finger=prev_finger,
                        prev_note=prev_note,
                        prev_is_black=prev_is_black,
                        curr_finger=curr_finger,
                        curr_note=curr_note,
                        curr_is_black=curr_is_black,
                        next_is_black=next_is_black_val,
                        prev_prev_finger=pp_finger,
                        prev_prev_note=pp_note,
                        prev_prev_is_black=pp_is_black,
                        hand=getattr(self, '_hand', 1),
                    )

                    # NetworkX allows multiple add_edge calls; keeps min weight via update
                    if G.has_edge(prev_node, curr_node):
                        if cost < G[prev_node][curr_node]['weight']:
                            G[prev_node][curr_node]['weight'] = cost
                    else:
                        G.add_edge(prev_node, curr_node, weight=cost)

        # -----------------------------------------------------------------
        # Connect final step nodes to sink
        # -----------------------------------------------------------------
        final_nodes = [n for n in G.nodes if isinstance(n, tuple) and len(n) == 3 and n[0] == target_len - 1]
        for node in final_nodes:
            G.add_edge(node, SINK, weight=0)

        # -----------------------------------------------------------------
        # Shortest path
        # -----------------------------------------------------------------
        try:
            path = nx.dijkstra_path(G, SOURCE, SINK, weight='weight')
        except nx.NetworkXNoPath:
            return []

        # Extract finger sequence from path (skip source and sink)
        finger_path = []
        for node in path:
            if isinstance(node, tuple) and len(node) == 3:
                _, _, finger = node
                if not finger_path or finger_path[-1] != finger or len(finger_path) < node[0] + 1:
                    # Only append the curr_finger once per step
                    if len(finger_path) <= node[0]:
                        finger_path.append(finger)

        return finger_path

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
