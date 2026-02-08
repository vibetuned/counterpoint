# Piano PASLCLRADE97 Rules

Based in the book "An Ergonomic Model of Keyboard Fingering for Melodic
Fragments"

## Span Level Definition

| Level | Type | Definition & Logic | Model Calculation / Constraint |
| :--- | :--- | :--- | :--- |
| **Practical** | **MaxPrac** | [cite_start]The maximum stretch a pianist actually uses in performance for finger legato[cite: 54]. | [cite_start]Smaller than anatomical limits ($MaxPoss$) to avoid awkward hand rotation[cite: 55, 56]. |
| **Practical** | **MinPrac** | [cite_start]The smallest permissible distance between consecutive notes[cite: 57]. | [cite_start]For thumb pairs, this is the maximum distance for passing under/over (negative number)[cite: 67]. |
| **Comfortable**| **MaxComf** | [cite_start]The upper limit of a "comfortable" stretch before strain points are applied[cite: 114]. | [cite_start]Set at exactly **2 semitones smaller** than $MaxPrac$ for all pairs[cite: 117]. |
| **Comfortable**| **MinComf** | [cite_start]The lower limit of a "comfortable" stretch[cite: 114]. | [cite_start]Set **2 semitones larger** than $MinPrac$ for thumb pairs; equal to $MinPrac$ for others[cite: 117]. |
| **Relaxed** | **MaxRel** | [cite_start]The interval where fingers fall without tension onto the keys[cite: 107]. | [cite_start]For non-thumb pairs, it is **twice the difference** between finger numbers[cite: 111]. |
| **Relaxed** | **MinRel** | [cite_start]The smallest interval of the "relaxed" or neutral hand range[cite: 110]. | [cite_start]Set at **1 semitone smaller** than $MaxRel$ for non-thumb pairs[cite: 111]. |


## Finger Span Values

| Finger Pair | MinPrac | MinComf | MinRel | MaxRel | MaxComf | MaxPrac |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1-2 | -5 | -3 | 1 | 5 | 8 | 10 |
| 1-3 | -4 | -2 | 3 | 7 | 10 | 12 |
| 1-4 | -3 | -1 | 5 | 9 | 12 | 14 |
| 1-5 | -1 | 1 | 7 | 11 | 13 | 15 |
| 2-3 | 1 | 1 | 1 | 2 | 3 | 5 |
| 2-4 | 1 | 1 | 3 | 4 | 5 | 7 |
| 2-5 | 2 | 2 | 5 | 6 | 8 | 10 |
| 3-4 | 1 | 1 | 1 | 2 | 2 | 4 |
| 3-5 | 1 | 1 | 3 | 4 | 5 | 7 |
| 4-5 | 1 | 1 | 1 | 2 | 3 | 5 |

## Rules

| Rule N° | Application | Description | Score | Source |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Consecutive Spans | [cite_start]**Stretch Rule**: Penalty for intervals that exceed the maximum comfortable span ($MaxComf$) or are less than the minimum comfortable span ($MinComf$)[cite: 215]. | [cite_start]2 points per semitone outside the range[cite: 215]. [cite_start]| [cite: 215] |
| 2 | Consecutive Spans | [cite_start]**Small-Span Rule**: Penalty when the span between two fingers is smaller than a relaxed span ($MinRel$), making the position feel cramped[cite: 226, 232]. | 1 point per semitone (if thumb involved); [cite_start]2 points per semitone (if thumb not involved)[cite: 226, 227]. [cite_start]| [cite: 226, 227, 232] |
| 3 | Consecutive Spans | [cite_start]**Large-Span Rule**: Penalty when spans exceed the maximum relaxed span ($MaxRel$)[cite: 256, 259]. | 1 point per semitone (if thumb involved); [cite_start]2 points per semitone (if thumb not involved)[cite: 256, 257]. [cite_start]| [cite: 256, 257, 259] |
| 4 | Next-to-Consecutive Spans | [cite_start]**Position-Change-Count Rule**: Penalty for changes in hand position, favoring fingerings that stay within a single position[cite: 320, 347]. | 2 points for a "full" change; [cite_start]1 point for a "half" change[cite: 320]. [cite_start]| [cite: 320, 347] |
| 5 | Next-to-Consecutive Spans | [cite_start]**Position-Change-Size Rule**: Penalty based on the physical distance the hand must travel during a position change[cite: 349, 370]. | [cite_start]1 point per semitone that the interval between the 1st and 3rd notes falls outside the $MinComf$ or $MaxComf$ range[cite: 349, 350]. [cite_start]| [cite: 349, 350, 370] |
| 6 | Finger Strength/Agility | [cite_start]**Weak-Finger Rule**: Penalty for using fingers considered less strong and agile (fingers 4 and 5)[cite: 419, 420]. | [cite_start]1 point for every use of finger 4 or finger 5[cite: 419]. [cite_start]| [cite: 419, 420] |
| 7 | Finger Strength/Agility | [cite_start]**Three-Four-Five Rule**: Penalty for using fingers 3, 4, and 5 consecutively, which are difficult to coordinate on the weak side of the hand[cite: 421, 422]. | [cite_start]1 point per occurrence of the sequence 3-4-5 (in any order)[cite: 421]. [cite_start]| [cite: 421, 422] |
| 8 | Finger Strength/Agility | [cite_start]**Three-to-Four Rule**: Penalty for following finger 3 immediately with finger 4, a difficult transition due to tendon limitations[cite: 440, 442]. | [cite_start]1 point per occurrence[cite: 440]. [cite_start]| [cite: 440, 442] |
| 9 | Finger Strength/Agility | [cite_start]**Four-on-Black Rule**: Penalty for a consecutive transition between fingers 3 and 4 where finger 3 is on white and finger 4 is on black[cite: 441, 444]. | [cite_start]1 point per occurrence (in either order)[cite: 441]. [cite_start]| [cite: 441, 444] |
| 10 | Black and White Keys | [cite_start]**Thumb-on-Black Rule**: Penalty for placing the thumb on a black key, which can displace the hand from a comfortable position[cite: 448, 468]. | [cite_start]1 base point + 2 points if the preceding note is white + 2 points if the following note is white[cite: 448, 449, 450]. [cite_start]| [cite: 448, 449, 450, 468] |
| 11 | Black and White Keys | [cite_start]**Five-on-Black Rule**: Penalty for placing the little finger on a black key when surrounded by white keys[cite: 451, 468]. | [cite_start]2 points if the preceding note is white + 2 points if the following note is white (0 points if both neighbors are black)[cite: 451, 452, 453]. [cite_start]| [cite: 451, 452, 453, 468] |
| 12 | Black and White Keys | [cite_start]**Thumb-Passing Rule**: Penalty based on the difficulty of passing the thumb under/over another finger relative to key elevation[cite: 493, 494]. | 1 point for passes on the same level; [cite_start]3 points for white-to-black passes involving the thumb[cite: 493]. [cite_start]| [cite: 493, 494] |