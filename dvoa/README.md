# Calculating DVOA For the Big Ten Conference

### Author: Jordan Betterman

### Email: <jordan.betterman@u.northwestern.edu>

#### DVOA Explained: <https://ftnfantasy.com/nfl/dvoa-explainer>

#### Link to Data: <https://collegefootballdata.com/exporter>

## Overview

Football Outsiders is a website that has calculated advanced statistics for NFL teams that better evaluate how good a team is on both sides of the ball. One advanced statistic is DVOA which stands for Defense-adjusted Value Over Average. This statistic shows how efficient a team or player is based on down and distance. For example, a 3-yard run on 1st and 10 and a 3-yard run on 3rd and 10 should not be compared as similar runs. These 3-yard runs are completely different because of the situation they occur in. This is what DVOA tries to tackle. How can we evaluate these 2 runs separately and combine these runs with the other plays in the game to evaluate players and teams effectively? Football Outsiders calculates this stat for the NFL but not for any college teams, so my task was to read the documentation Football Outsiders gives to the public and recalculate DVOA for the Big Ten.

## How To Calcualte DVOA

The foundation of calculating DVOA is identifying if a play was a success or not. The way that this is done is by awarding **success points** to plays that gain a certain number of yards based on what down it is. 1 success point is awarded if the percentage of yards gained to a first down. If a play results in a first down and gains more than 10 yards, then bonus points are awarded. Finally, if a play results in a loss of yards, interception, or fumble, then penalties are given to that play. A fumble is awarded success points a range between 2 values depending on the field position of where the fumble occurred. A penalty is given regardless of who recovers the fumble. The success point values are shown below:

<center><table>
<tr><th style="text-align: center">1 Success Point</th><th style="text-align: center">Bonus Points</th><th style="text-align: center">Penalties</th></tr>
<tr><td>

| Down | Yards Gained % |
| :--: |      :---:     |
|  1   |  More than 45% |
|  2   |  More than 60% |
| 3 & 4|  More than 100%|
</td><td>

| Yards Gained | Success Points |
|    :--:      |      :---:     |
|  10 - 19     |        3       |
|  20 - 39     |        4       |
|     40+      |        5       |

</td><td>

| Event | Success Points |
|  :--: |   :---:     |
|  Loss of 3 or more yards |       -1       |
|      Interception            |       -6       |
|       Fumble                 |   -1.4 to -4   |

</td></tr> </table></center>

Once success points are awarded, we will find the conference average of success points for each down and distance. Then we will calculate the VOA in DVOA by dividing the success points awarded for that play by the conference average of success points awarded in that situation. Once this is completed, the final step is to add the defense adjustment to the calculation. This is done by subtracting the calculated VOA from the success percentage of the opposition (gaining success points on offense or not allowing success points on defense). After that step is done, we will normalize the results so the conference average DVOA is 0%. 

## How can we use DVOA?
DVOA is a great statistic that can evaluate where teams lie within the Big Ten conference as well as within the FBS. A huge advantage to how DVOA is calculated is that we can look at certain down and distance scenarios to identify where teams struggle or excel on both sides of the ball. This can save a lot of time when scouting an opponent so the coaching staff can focus on game planning around those scenarios. Obviously, this is **NOT** a replacement for film study, but it can help guide where coaches look when scouting opponents. Another use case for DVOA is that we can evaluate player performance compared to the Big Ten or FBS. Calculating DVOA for offensive skill players is pretty easy to do, but if we wanted to look at evaluating O-Linemen or defensive players it can get difficult. So, it would take some time to thoughtfully calculate DVOA for non-skill players in the future.

<div style="page-break-after: always;"></div>

## Results
I calculated the DVOA for each Big Ten team and then split it up into overall, run, and pass for both offense and defense. These numbers are normalized meaning that the best offenses have a positive DVOA while the worst offenses have a negative DVOA. This is then flipped for defense where the best defenses have a negative DVOA and the worst defenses have a positive DVOA. If a team has a 0 DVOA then they are at the conference average. Now lets look at the results for offensive DVOA:

<center><table>
<tr><th style="text-align: center">Overall Offense</th><th style="text-align: center">Passing Offense</th><th style="text-align: center">Rushing Offense</th></tr>
<tr><td>

| Team | DVOA |
| :--: |  :--:|
|  Ohio State |  10.85%  |
|  Minnesota  |  7.92%  |
|  Michigan |    7.88%  |
| Penn State| 2.37% |
| Wisconsin | 2.04%  |
| Michigan State | 0.23% |
| Illinois | 0.01% |
| Purdue | -0.08% |
| Maryland | -2.50% |
| Rutgers | -5.22% |
| Indiana | -5.26% |
| Northwestern | -6.47% |
| Iowa | -6.81% |
| Nebraska | -7.65% |

</td><td>

| Team | DVOA |
| :--: |  :--:|
|  Ohio State |  7.19%  |
|  Minnesota  |  6.20%  |
|  Wisconsin |    5.29%  |
| Penn State| 4.28% |
| Purdue | 2.46%  |
| Illinois | 1.80% |
| Michigan State | 1.40% |
| Michigan | -0.02% |
| Maryland | -1.56% |
| Iowa | -3.43% |
| Northwestern | -3.72% |
| Rutgers | -4.06% |
| Nebraska | -6.43% |
| Indiana | -6.53% |

</td><td>

| Team | DVOA |
| :--: |  :--:|
|  Michigan |  19.59%  |
|  Ohio State  |  18.12%  |
|  Minnesota |    12.55%  |
| Penn State| -0.05% |
| Wisconsin | -1.30%  |
| Illinois | -1.52% |
| Indiana | 2.91% |
| Michigan State | -3.21% |
| Maryland | -4.80% |
| Purdue | -7.02% |
| Rutgers | -7.70% |
| Nebraska | -10.63% |
| Northwestern | -11.46% |
| Iowa | -13.39% |

</td></tr> </table> </center>

One interesting takeaway from the offensive results is the Pass Offense of Wisconsin is ranked as the 3rd most efficient pass offense in the conference. Although Wisconsin did not pass the ball a lot last season, they were efficient in getting the yards needed given the situation. Another takeaway is that there was a large left skewness in rushing offense DVOA. This means that either the top 3 (Michigan, Ohio State, and Minnesota) were a lot better at running the ball compared to the rest of the conference or the bottom teams were a lot worse at running the ball compared to the rest of the conference. My conclusion for this phenomenon is that the top 3 teams were way better than the conference in rushing. The top 3 teams in rushing DVOA averaged over 2 rushing touchdowns and over 200 rushing yards per game while the rest of the conference was not as nearly as close to the top 3 in those categories. Now lets look at the results from defensive DVOA.

<center><table>
<tr><th style="text-align: center">Overall Defense</th><th style="text-align: center">Passing Defense</th><th style="text-align: center">Rushing Defense</th></tr>
<tr><td>

| Team | DVOA |
| :--: |  :--:|
|  Illinois |  -6.89%  |
|  Michigan  |  -5.23%  |
|  Ohio State |    -3.58%  |
| Iowa | -3.08% |
| Wisconsin | -2.65%  |
| Minnesota | -1.94% |
| Penn State | -0.08% |
| Purdue | 0% |
| Rutgers | 0.07% |
| Indiana | 2.59% |
| Maryland | 2.60% |
| Nebraska | 3.97% |
| Northwestern | 5.08% |
| Michigan State | -7.65% |

</td><td>

| Team | DVOA |
| :--: |  :--:|
|  Rutgers |  -7.65%  |
|  Michigan  |  -5.51%  |
|  Iowa |    -3.86%  |
| Illinois | -3.04% |
| Penn State | -1.88%  |
| Nebraska | -1.00% |
| Ohio State | -0.07% |
| Minnesota | 0.02% |
| Maryland | 1.41% |
| Northwestern | 2.91% |
| Wisconsin | 3.48% |
| Purdue | 4.20% |
| Michigan State | 4.23% |
| Indiana | 7.84% |

</td><td>

| Team | DVOA |
| :--: |  :--:|
|  Illinois |  -15.15%  |
|  Wisconsin  |  -12.50%  |
|  Ohio State |    -8.58%  |
|  Michigan | -8.52% |
|  Purdue | -7.67%  |
|  Minnesota | -6.13% |
|  Iowa | -4.87% |
|  Indiana | -3.71% |
|  Penn State | 1.49% |
|  Maryland | 4.25% |
|  Michigan State | 9.12% |
|  Northwestern | 11.20% |
|  Rutgers | 13.64% |
|  Nebraska | 14.76% |

</td></tr> </table> </center>

The main takeaway from this table is that Minnesota sized up to be around the conference in defense efficiency. This is an interesting find from these results because Minnesota was 3rd in points allowed, 5th in rushing yards allowed per game, and 4th in passing yards allowed per game. I believe Minnesota is in this position because they did not produce as many turnovers compared to the rest of the conference. Turnovers give larger penalties than tackles for loss or sacks. This means that teams with more turnovers per game are higher ranked in Defensive DVOA rankings. Illinois, for example, is ranked 1st in overall defensive DVOA and also was 1st in the Big Ten in turnovers per game (2.5).

<div style="page-break-after: always;"></div>

## Conclusion
Overall, I think this statistic will be very helpful for evaluating teams this season. I think it will be helpful to the coaching staff because it will give a good indication of how efficient teams are in certain facets of the game as well as identifying weak areas in their game compared to the conference. Now that this research work has been completed, I can easily transfer this code over to datasets that provide down, distance, and play result. So, scaling up this project will be simple to do. I have calculated all the Down and Distance DVOAs for the Big Ten, but there is a lot of data to go through for each team. Therefore, I left it out of this report to show the general application of DVOA. 

## Next Steps

Now that we have calculated DVOA successfully. I want to further expand calculating DVOA to all FBS teams. That way we can evaluate teams and players at both their conference level and at the entire FBS level. I would also like to create an SQL database where the calculations can be housed, time-stamped by date, and easily accessible. That way we can identify trends for the teams that we face each week and see where they may be improving or struggling in each week. As mentioned in the conclusion, I have written the DVOA calculation to be to simple to scale up in size to calculate all FBS teams. The issue that I will tackle next is to figure out how to store the data in a secure and accessible way.