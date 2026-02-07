Constraints 3 PROBLEM DESCRIPTION Table 2: Set of rules composing the objective function (adapted from Parncutt et al. (1997), Jacobs (2001) and Al Kasimi et al. (2007)). 

| Rule | Application | Description | Score | Source |
| --- | --- | --- | --- | --- |
| 1 | All | For every unit the  distance between two consecutive notes is below MinComf or exceeds MaxComf. | +2 | Parncutt et al. |
| 2 | All | For every unit the distance between two consecutive notes is below MinRel or exceeds MaxRel. | +1 | Parncutt et al. and Jacobs |
| 3 | Monophonic | For three consecutive notes: If the distance between a first and third note is below MinComf or exceeds MaxComf: add one point. In addition to that, if the pitch of the second note is between the other two pitches, is played by the thumb and the distance between the first and third note is below MinPrac or exceeds MaxPrac: add another point. Finally, if the first and third note have the same pitch, but are played by a different finger: add another point. | +1 +1 +1 | Parncutt et al. |
| 4 | Monophonic | For every unit the distance between a first and third note is below MinComf or exceeds MaxComf. | +1 | Parncutt et al. |
| 5 | Monophonic | For every use of the fourth finger. | +1 | Parncutt et al. and Jacobs |
| 6 | Monophonic | For the use of the third and the fourth finger consecutively. | +1 | Parncutt et al. |
| 7 | Monophonic | For the use of the third finger on a white key and the fourth finger on a black key consecutively in any order. | +1 | Parncutt et al. |
| 8 | Monophonic | When the thumb plays a black key: add a half point. Add one more point for a different finger used on a white key just before and one extra for one just after the thumb. | +0.5 +1 +1 | Parncutt et al. |
| 9 | Monophonic | When the fifth finger plays a black key: add zero points. Add one more point for a different finger used on a white key just before and one extra for one just after the fifth finger. | +1 +1 | Parncutt et al. |
| 10 | Monophonic | For a thumb crossing or passing another finger on the same level (white–white or black–black). | +1 | Parncutt et al. |
| 11 | Monophonic | For a thumb on a black key crossed by a different finger on a white key. | +2 | Parncutt et al. |
| 12 | Monophonic | For a different first and third consecutive note, played by the same finger, and the second pitch being the middle one. | +1 | Own rule |
| 13 | All | For every unit where the distance between two following notes is below MinPrac or exceeds MaxPrac. | +10 | Own rule, based on constraint of Parncutt et al. |
| 14 | Polyphonic | Apply rules 1, 2 (both with doubled scores) and 13 within one chord. | +10 | Own rule, based on Al Kasimi et al. |
| 15 | All | For consecutive slices containing exactly the same notes (with identical pitches), played by different fingers. | +1 | Own rule |