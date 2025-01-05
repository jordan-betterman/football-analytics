# Fourth Down Decision Making

#### **Author: Jordan Betterman**
#### **Email: jordan.betterman@u.northwestern.edu**
#### Link to Database Used: [CollegeFootballData.com](https://collegefootballdata.com/exporter)

## Overview
Recently in the analytics world, data scientists have tried to tackle when to punt, kick, or go for it on 4th down. Most of these attempts have gotten close but are missing some key concepts and variables that can steer their deicion making to be overly aggressive in certain situations.

## Purpose
The purpose of this project is to take existing fourth down decision making models and modify them to take realistic in-game scenarios into account when making a prediction.

## The Problem
From my research, most existing fourth down decision making models can give an overly aggressive suggesstion given the in-game scenario. For example, Team A could be leading by 35 against Team B, it's 4th and 5 on the 45 yard line with 5 minutes left in the game. The obvious decision for this situation is to punt or kick the ball, but the model would predict to go for it. The reason why the model would decide to go for it in this situation is that it doesb't take time remaining in the game into consideration. Some researchers have even written in their papers that they **"assume an infinite game"**. That's a huge mistake! It shouldn't matter how much data there is to gather for calculating EPA because it throws away late game nuances that aren't relevant earlier in the game.

## The Solution
The solution to this problem is adding time into the equation for calculating EPA. The way I tackled this issue is I translated the time data for football (quarter, minutes left) into soccer's version of time data (total minutes played). For example, **"12 minutes left in the 1st quarter"** translates into **3rd minute of play**. The reason I translated time to a soccer time format because it makes it easy to categorize the data by time intervals while taking quarter into account instead splitting the dataset by quarter. I categorized data into 3 minute time intervals because it would then lead to looking into endgame situations simpler later on in the process.


## Model Making Results
Below are the results of the 4 combinations of preprocessing methods. There were 2 EPA models that I evaluated. One EPA calculation method was subtracting the Expected Points (EP) from the actual points scored on the play. The other EPA calculation method was subtracting the EP before the play by the EP after the play was completed. Then I also split up trying categorical and numerical columns for time and field position. The advantage of intervalling time and field position is that there will be more plays that fall inside a category instead of a single numerical value. For example, there will be more plays that count for a time interval between 15-13 minutes compared to the number of plays at the 15 minute mark alone.

Here is a table with the r^2 values for the 3 models I made (field goal, go for it, or punt):

| Preprocessing Method | Field Goal | Punt | Go For It  |
| -------------------- |   :---:    | :---: |   :---:   |
| Actual-EP Categorical |   0.47    | 0.03  |     0     |
| Actual-EP Numerical  |   0.40     |  0    |     0     |
| EPB-EPA Categorical  |    0.68    | 0.40  |    0.47   |
| EPB-EPA Numerical    |    0.67    | 0.46  |    0.40   |

As you can see from the table above, the EP Before - EP After using categorical intervals was the best performing model. I think it's the best because it helps give us a more generalized approach to deciding what to do on 4th down. Although the numerical approach did show good numbers compared to the other models that were ran, we don't need to have the exact yard line and the exact time interval. We can give ranges for where the ball is located and what the game clock situation is. The difference between EPA from one minute to the next can be marginal which means we wouldn't gain much information based on what may happen a minute later. There is a caveat with this logic. We would have to look at splitting up the time intervals in to more reasonable buckets. We can give more focus to later in the game where the could be a huge difference in EPA.

## Final Results
After some tweaking of time intervals, I processed the data and predicted all 4th down decisions from the past 5 years of big ten play. In the end there was **A LOT** of data to go through, so I narrowed down the situations to look towards reasonable situations. I looked into sitautions where the yards to go was less than or equal to 10 yards, any field position within the opponents half of the field, and the last 5 minutes in the first and second half. I then filtered it down one more time to a short, mid, and long distance.

- Short: 1-3 yards to the first
- Mid: 4-6 yards to the first
- Long: 7-10 yards to the first

The results show only a few situations where going for it was the best option. Here are the following situations:

- Short
    - 1-5 yard line down 14
    - 1-5 yard line down 7
    - 1-5 yard line tied
    - 1-5 yard line up 3
    - 1-5 yard line up 10
- Mid
    - 1-5 yard line down 7
    - 1-5 yard line up 3
    - 1-5 yard line up 14
- Long
    - 26-30 yard line down 3
    - 31-35 yard line down 9

The rest of the situations within the opponent's half were predicted as a Field Goal Attempt as the best option. But in hindsight all these situations mentioned above make sense. Being within the 5 yard line is already a high probabilty to succeed in going for it. But it also helps if we turn it over on downs giving the opposing team tough field position.

## Conclusion 
Overall, I think this research was a success. The models that I made were not overly aggressive as well as give good context on how these 4th decisions should be in Big Ten Conference play. Some possible further would be to look into adding the new teams joining the Big Ten into the equation and see if that changes anything. 

