# Predicting Play Characteristics Using Presnap Data

## Overview
The first project I worked on was predicting whether a team will run or pass the ball based on presnap variables that are useful in making a defensive play call. The way you guys do this right now is down breaks. Although that information is useful, there are possible combinations of personnel packges, downs, and distances that a team hasn't played in-game before. This leaves the coaching staff to make an ill-informed decision for that scenario because they don't have the data beforehand to plan on. Therefore, I wanted to write program that can help speed up creating down breaks and also predict scenarios that a team hasn't played yet.

## Variables Used to Predict Run/Pass:
- Quarter
- Minutes left
- Down
- Distance
- Personnel Package
- Field Position
- Score Differential
- Previous Play


## Results
The results of this project was a astounding success! The cross validation accuracy scores for the models hovered around 70%-80%. That's really good! The time it takes to run one model and predict all the data is less than 10 minutes. Also, the script can predict other categorical data columns that PFF provides. I have expanded the capabilities of the script to also predict pass direction, whether a play will be play action or not, run concepts, and run direction. Pass direction was the least accurate, but the rest of the variables predicted had similar accurate scores to the run/pass models that were created. Overall, this project was a success. The next step is taking the prediction set that the model created and communicating it to the coaching staff effectively.


### Used PFF Data as Model Input