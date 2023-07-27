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


### Link to Database Used: [CollegeFootballData.com](https://collegefootballdata.com/exporter)