# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import logging as logger
from time import time
from arguments import Arguments
from pathlib import Path

logger.getLogger().setLevel(logger.INFO)


def cleaner(dataframe, target):
    dataframe["Gainloss"] = dataframe["Gainloss"].replace(" ", np.nan, regex=True)

    # created a new column that moved the yards gained or lost on a play down a row
    dataframe["Yards on Previous Play"] = dataframe["Gainloss"].shift()
    dataframe["previous_play"] = dataframe["Runpass"].shift()

    dataframe = dataframe[(dataframe["Runpass"] == "R") | (dataframe["Runpass"] == "P")]
    dataframe = dataframe[
        (dataframe["previous_play"] == "R") | (dataframe["previous_play"] == "P")
    ]

    # extra data cleaning pieces to make downs be 1-4, runpass to only be R or P, and offensive personnel to be known and have 11 players on the field
    if target == "Runpass" or target == "Playaction":
        dataframe = dataframe[
            (dataframe["Offpersonnelbasic"] != "Unknown")
            & (dataframe["Offpersonnelbasic"] != "10 Men")
        ]
    elif target == "Passdirection":
        dataframe = dataframe[
            (dataframe["Offpersonnelbasic"] != "Unknown")
            & (dataframe["Offpersonnelbasic"] != "10 Men")
            & (dataframe["Runpass"] == "P")
        ]
    elif target == "Rbdirection" or target == "Runconceptprimary":
        dataframe = dataframe[
            (dataframe["Offpersonnelbasic"] != "Unknown")
            & (dataframe["Offpersonnelbasic"] != "10 Men")
            & (dataframe["Runpass"] == "R")
        ]
        if target == "Runconceptprimary":
            dataframe = dataframe[(dataframe["Runconceptprimary"] != "UNDEFINED")]
    elif target == "Predictions":
        dataframe = dataframe[
            (dataframe["Offpersonnelbasic"] != "Unknown")
            & (dataframe["Offpersonnelbasic"] != "10 Men")
        ]
        subset = dataframe[
            [
                "Quarter",
                "Minutes Left",
                "Down",
                "Distance",
                "Fieldposition",
                "Offpersonnelbasic",
                "Yards on Previous Play",
                "Scoredifferential",
                "previous_play",
            ]
        ]
        return subset
        # created a subset with the features that can be seen presnap to do ML on
    subset = dataframe[
        [
            "Quarter",
            "Minutes Left",
            "Down",
            "Distance",
            "Fieldposition",
            "Offpersonnelbasic",
            target,
            "Yards on Previous Play",
            "Scoredifferential",
            "previous_play",
        ]
    ]
    subset = subset.dropna(
        subset=[
            "Quarter",
            "Minutes Left",
            "Down",
            "Distance",
            "Fieldposition",
            "Offpersonnelbasic",
            target,
            "Scoredifferential",
            "Yards on Previous Play",
            "previous_play",
        ]
    )

    return subset


def prediction_set(dataset):
    quarter = range(1, 5)
    down = range(1, 4)
    distance = range(1, 11)
    minutes_left = range(1, 16)
    personnel = dataset["Offpersonnelbasic"].unique().tolist()
    previous_play = ["R", "P"]
    score_differential = range(-14, 14, 7)
    yards_on_previous_play = [-10, -5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    field_position = [-30, -35, -40, -45, 50, 45, 40, 35, 30]

    master = {
        "Quarter": [],
        "Minutes Left": [],
        "Down": [],
        "Distance": [],
        "Offpersonnelbasic": [],
        "Yards on Previous Play": [],
        "Fieldposition": [],
        "Scoredifferential": [],
        "previous_play": [],
    }

    for i in quarter:
        for m in minutes_left:
            for d in down:
                for l in distance:
                    for p in personnel:
                        for f in field_position:
                            for s in score_differential:
                                for y in yards_on_previous_play:
                                    for play in previous_play:
                                        master["Quarter"].append(i)
                                        master["Minutes Left"].append(m)
                                        master["Down"].append(d)
                                        master["Distance"].append(l)
                                        master["Offpersonnelbasic"].append(p)
                                        master["Yards on Previous Play"].append(y)
                                        master["Fieldposition"].append(f)
                                        master["Scoredifferential"].append(s)
                                        master["previous_play"].append(play)

    final_test = pd.DataFrame(master)

    return final_test


def find_max(param):
    max_index = 0
    agg = 0
    for i in param:
        if i > param[max_index]:
            max_index = agg
        agg += 1
    return max_index


def download_data(dataset, name):
    output_file_path = Path(name)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_file_path)


def randomforest_auto(
    file_path: str, target: str, output: str, team: str, year: str
) -> pd.DataFrame:
    start = time()
    # call cleaner data to clean the dataset for ML use
    dataframe = pd.read_csv(file_path)

    predictions = prediction_set(cleaner(dataframe, "Predictions"))
    logger.info("Preiction set completed")

    cleaned_data = cleaner(dataframe, target)

    cleaned_data = cleaned_data.astype({"Yards on Previous Play": "int", 'Minutes Left': "int"})

    cleaned_data_w_dummies = pd.get_dummies(cleaned_data, columns=["previous_play"])
    logger.info(f"completed data cleaning with dummies")

    # predictor variables used: all variables besides the target variable
    predictors = cleaned_data_w_dummies.columns.drop([target, "Offpersonnelbasic"])
    target = target  # target variable

    # splits the subset into a training set to fit the models on and a testing set to test the models on for their accuracy
    train_data, test_data, train_sln, test_sln = train_test_split(
        cleaned_data_w_dummies[predictors],
        cleaned_data_w_dummies[target],
        test_size=0.2,
        random_state=0,
    )

    estimators = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

    criterions = ["gini", "entropy"]

    results = {
        "max_depth": [],
        "n_estimators": [],
        "min_samples_split": [],
        "min_samples_leaf": [],
        "max_features": [],
        "criterion": [],
    }  # house all the accuracies when tuning parameters

    params = {
        "max_depth": 0,
        "n_estimators": 0,
        "min_samples_split": 0,
        "min_samples_leaf": 0,
        "max_features": 0,
        "criterion": "",
    }  # house the paramters for the model

    max_accuracy = 0

    # finding the optimal depth for the decision trees to go to
    for i in range(1, 30):
        forest = RandomForestClassifier(max_depth=i, random_state=0)
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = accuracy_score(test_sln, prediction)
        results["max_depth"].append(val)

    max_accuracy = max(results["max_depth"])
    logger.info(
        f"Best Accuracy after tuning max depth is: {max(results['max_depth'])}"
    )  # print max accuracy to see if it improves
    depth = find_max(results["max_depth"]) + 1
    params["max_depth"] = depth

    # find the optimal amount of decision trees to use
    for i in estimators:
        forest = RandomForestClassifier(max_depth=depth, n_estimators=i, random_state=0)
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = accuracy_score(test_sln, prediction)
        results["n_estimators"].append(val)

    if max(results["n_estimators"]) < max_accuracy:
        logger.info(f"No improvement made. N Estimators given default value")
        max_estimators = 100
        params["n_estimators"] = estimators[max_estimators]
    else:
        logger.info(
            f"Best Accuracy after tuning n_estimators is: {max(results['n_estimators'])}"
        )  # print max accuracy to see if it improves
        max_accuracy = max(results["n_estimators"])
        max_estimators = find_max(results["n_estimators"])
        params["n_estimators"] = estimators[max_estimators]

    # find the optimal amount of sample splits to use
    for i in range(2, 15):
        forest = RandomForestClassifier(
            max_depth=depth,
            n_estimators=estimators[max_estimators],
            min_samples_split=i,
            random_state=0,
        )
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = accuracy_score(test_sln, prediction)
        results["min_samples_split"].append(val)

    if max(results["min_samples_split"]) < max_accuracy:
        logger.info(f"No improvement made. Min Samples Split given default value")
        max_samp_split = 2
        params["min_samples_split"] = 2
    else:
        logger.info(
            f"Best Accuracy after tuning min samples split is: {max(results['min_samples_split'])}"
        )  # print max accuracy to see if it improves
        max_accuracy = max(results["min_samples_split"])
        max_samp_split = find_max(results["min_samples_split"]) + 2
        params["min_samples_split"] = max_samp_split

    # find the optimal amount of sample leafs to use
    for i in range(1, 15):
        forest = RandomForestClassifier(
            max_depth=depth,
            n_estimators=estimators[max_estimators],
            min_samples_split=max_samp_split,
            min_samples_leaf=i,
            random_state=0,
        )
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = accuracy_score(test_sln, prediction)
        results["min_samples_leaf"].append(val)

    if max(results["min_samples_leaf"]) < max(results["min_samples_split"]):
        logger.info(f"No improvement made. Min Samples Leaf given default value")
        max_samp_leaf = 1
        params["min_samples_leaf"] = 1
    else:
        logger.info(
            f"Best Accuracy after tuning min samples leaf is: {max(results['min_samples_leaf'])}"
        )  # print max accuracy to see if it improves
        max_accuracy = max(results["min_samples_leaf"])
        max_samp_leaf = find_max(results["min_samples_leaf"]) + 1
        params["min_samples_leaf"] = max_samp_leaf

    # find the optimal amount of features to use
    for i in range(1, len(cleaned_data.columns)):
        forest = RandomForestClassifier(
            max_depth=depth,
            n_estimators=estimators[max_estimators],
            min_samples_split=max_samp_split,
            min_samples_leaf=max_samp_leaf,
            max_features=i,
            random_state=0,
        )

        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = accuracy_score(test_sln, prediction)
        results["max_features"].append(val)

    if max(results["max_features"]) < max_accuracy:
        logger.info(f"No improvement made. Min Samples Leaf given default value")
        max_feat = len(cleaned_data.columns) ** 0.5
        params["max_features"] = max_feat
    else:
        logger.info(
            f"Best Accuracy after tuning max features is: {max(results['max_features'])}"
        )  # print max accuracy to see if it improves
        max_accuracy = max(results["max_features"])
        max_feat = find_max(results["max_features"]) + 1
        params["max_features"] = max_feat

    # find the optimal criterion
    for i in criterions:
        forest = RandomForestClassifier(
            max_depth=depth,
            n_estimators=estimators[max_estimators],
            min_samples_split=max_samp_split,
            min_samples_leaf=max_samp_leaf,
            max_features=max_feat,
            criterion=i,
            random_state=0,
        )

        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = accuracy_score(test_sln, prediction)
        results["criterion"].append(val)

    logger.info(
        f"Best Accuracy after tuning criterion is: {max(results['criterion'])}"
    )  # print max accuracy to see if it improves
    max_accuracy = max(results["criterion"])
    max_criterion = find_max(results["criterion"])
    params["criterion"] = criterions[max_criterion]

    logger.info(f"Final Model Accuracy { max_accuracy}")

    logger.info(f"Tuned parameters: {params}")

    logger.info(f"Random Forest Feature Importance {forest.feature_importances_}")

    ######################################################
    # create the prediction set and predict values from it#
    ######################################################

    final_test = predictions

    final_test_dummy = pd.get_dummies(
        final_test, columns=["Offpersonnelbasic", "previous_play"]
    )
    final_test_data = final_test_dummy[predictors]

    forest = RandomForestClassifier(
        max_depth=depth,
        n_estimators=estimators[max_estimators],
        min_samples_split=max_samp_split,
        min_samples_leaf=max_samp_leaf,
        max_features=max_feat,
        criterion=criterions[max_criterion],
        random_state=0,
    )
    forest.fit(train_data, train_sln)
    predictions = forest.predict(final_test_data)
    logger.info('completed predicting')

    final_test[f"Predicted {target}"] = predictions

    grouped = final_test.groupby(
        ["Offpersonnelbasic", "Down", "Distance", "Minutes Left", f"Predicted {target}"]
    )[f"Predicted {target}"].agg(['count']).reset_index()
    logger.info(grouped.head())

    grouped['percentage'] = grouped['count'] / grouped.groupby(["Offpersonnelbasic", "Down", "Distance", "Minutes Left"])["count"].transform('sum')
    logger.info(grouped.head())

    download_data(grouped, f"{output}/{team}/{target}_{year}.csv")

    logger.info(f"Time Duration: {(time() - start)/60} minutes")
    return grouped


def main():
    args = Arguments().parse()
    logger.info(args)

    randomforest_auto(args.file, args.predictor, args.output, args.team, args.year)


main()
