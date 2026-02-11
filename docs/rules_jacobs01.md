# Piano Jacobs 2001 Rules

Based on Refinements to the Ergonomic Model for Keyboard Fingering of Parncutt, Sloboda,
Clarke, Raekallio, and Desain (2001)

## Implementation Logic

Jacobs defines the physical distance ranges based on the average inter-key distance of a standard keyboard where the octave width is 165 mm, The model still use the numbers from Parncutt's Table: Finger Span Values. The logic follows these steps:

1. Measure Actual Physical Distance ($D$): Calculate the distance in millimeters between the centers of the two keys involved in the interval.
2. Define Reference Distances ($d_i$): Let $d_i$ be the average physical distance of a musical interval of $i$ semitones (approximately $13.7 \text{ mm} \times i$).
3. Establish Boundaries ($B_i$): Create "bins" or ranges for each distance.
    1. The lower boundary of range $i$ is $(d_i + d_{i-1}) / 2$.
    2. The upper boundary of range $i$ is $(d_{i+1} + d_i) / 2$.
4. Assign the Index: If the actual physical distance $D$ falls within the boundary $B_i$, the interval is assigned the value $i$ for all subsequent rule calculations (Stretch, Small-Span, Large-Span).

## Rules

The rules are the same as in Parncutt's model, but with some modifications and additional rules:

| Rule N° | Application | Description | Score | Source |
| :--- | :--- | :--- | :--- | :--- |
| A | Span Logic | **Physical Distance Mapping**: Replaces semitones with physical distance ranges in mm. [cite_start]This ensures that physically identical spans (like E-F and C-D) are treated the same regardless of musical interval[cite: 1188, 1196, 1197]. | [cite_start]Re-maps intervals to a "semitone size" based on physical inter-key distance[cite: 1196]. | Jacobs (2001) |
| B | Black Keys | [cite_start]**Modified Thumb/Five on Black**: While maintaining the spirit of the rule, Jacobs emphasizes these are "short-finger-on-black" rules that depend heavily on the elevation context of neighboring keys[cite: 1215]. | [cite_start]Context-dependent (retains Parncutt's scores but re-justifies via ergonomics)[cite: 1160, 1215]. | Jacobs (2001) |
| C | Large Spans | **Unified Large-Span Rule**: Removes the extra penalty for non-thumb pairs. [cite_start]Jacobs argues that stretching non-thumb fingers is not inherently more difficult than stretching the thumb[cite: 1204, 1215]. | [cite_start]1 point per semitone exceeding MaxRel (for ALL finger pairs)[cite: 1215]. | Jacobs (2001) |
| 6 (Mod) | Strength | **Modified Weak-Finger Rule**: Removed the penalty for the fifth finger (5). [cite_start]Modern pedagogy recognizes the 5th finger as strong due to specialized outer-hand muscles[cite: 1200, 1201]. | [cite_start]Only penalizes finger 4 (1 point per use)[cite: 1202]. | Jacobs (2001) |
| 7 (Disc) | Coordination | [cite_start]**Three-Four-Five Rule (DISABLED)**: Jacobs argued this rule was redundant as other rules (like 3-to-4 transitions) already account for the coordination difficulty. | [cite_start]0 points (Disabled)[cite: 1202]. | Jacobs (2001) |
| 10/11 | Technique | [cite_start]**Short-Finger Context**: Re-affirms that 1 and 5 on black keys are mostly difficult because they displace the hand distally "into the keys"[cite: 1167]. | (Kept Parncutt scores but refined the ergonomic logic) [cite_start][cite: 1215, 1266]. | Jacobs (2001) |