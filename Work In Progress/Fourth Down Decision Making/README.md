# Fourth Down Decision Making

#### **PROJECT IS WORK IN PROGRESS**

## Overview
Recently in the analytics world, data scientists have tried to tackle how to decide whether to punt, kick, or go for it on 4th down. Most of these attempts have gotten close but are missing some key concepts and variables that can steer their deicion making to be overly aggressive in certain situations.

## Purpose
The purpose of this project is to take existing fourth down decision making models and modify them to take the in-game scenario into account when making a prediction.

## The Problem
From my research, most existing fourth down decision making models (and Expected Points Added models) can give an overly aggressive given the in-game scenario. For example, Team A could be leading by 35 against Team B, it's 4th and 5 on the 45 yard line with 5 minutes left in the game. The obvious decision for this situation is to punt the ball, but the model would predict to go for it. The reason why the model would decide to go for it in this situation is that it use time remaining in the game. Some researchers have even written in their papers that they **"assume an infinite game"**. That's a huge mistake! It shouldn't matter how much data there is to gather for calculating EPA because it throws away late game nuances that aren't relevant earlier in the game.

## The Solution
The solution to this problem is adding time into the equation for calculating EPA. The way I tackled this issue is I translated the time data for football (quarter, minutes left) into soccer's version of time data (total minutes played). For example, **"12 minutes left in the 1st quarter"** translates into **3rd minute of play**. The reason I translated time to a soccer time format because it makes it easy to categorize the data by time intervals while taking quarter into account instead splitting the dataset by quarter. Initially I categorized data into 3 minute time intervals. I thought this would be a good starting point when making initial models, but I am open to suggestions on what a good spread could look at.


## The Results
Below are the results of the 4 combinations of preprocessing methods. There were 2 EPA models that I evaluated. One EPA calculation method was subtracting the Expected Points (EP) from the actual points scored on the play. The other EPA calculation method was subtracting the EP before the play by the EP after the play completed. Then I also split up trying categorical and numerical columns for time and field position. The advantage of intervalling and categorizing is that there will be more plays that fall inside a category instead of a single numerical value. For example, there will be more plays that count for a time interval between 15-13 minutes compared to the number of plays at the 15 minute mark alone.

Here is a table with the r2 values for the 3 models I made (field goal, go for it, or punt):

| Preprocessing Method | Field Goal | Punt | Go For It  |
| -------------------- |   :---:    | :---: |   :---:   |
| Actual-EP Categorical |   0.47    | 0.03  |     0     |
| Actual-EP Numerical  |   0.40     |  0    |     0     |
| EPB-EPA Categorical  |    0.68    | 0.40  |    0.47   |
| EPB-EPA Numerical    |    0.67    | 0.46  |    0.40   |

As you can see from the table above, the EP Before - EP After using categorical intervals was the best performing model. I think it's the best because it helps give us a more generalized approach to deciding what to do on 4th down. Although the numerical approach did show good numbers compared to the other models that were ran, we don't need to have the exact yard line and the exact time interval. We can give ranges for where the ball is located and what the game clock situation is. The difference between EPA from one minute to the next could be very marginal which means we wouldn't gain much information based on what may happen a minute later. There is a caveat with this logic. We would have to look at splitting up the time intervals in to more reasonable buckets. We can then give more focus to later in the game where the could be a huge difference in EPA between 3 minutes or 2 minutes remaining in the game.

## Next Steps
The next steps of this project is to look deeper into splitting up the time intervals to weigh the end of the game better. After that has been completed, we will run the model and predict the values in the data I have pulled. Then, we will see which EPA value for the 3 decisions is the highest and assign a value to it. Finally, we will download it into a spreadsheet and interpret the results.

### Link to Database Used: [CollegeFootballData.com](https://collegefootballdata.com/exporter)