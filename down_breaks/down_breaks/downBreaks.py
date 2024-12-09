# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from itertools import product
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

    if len(dataframe["Offpersonnelbasic"].unique()) > 10:
        personnel_packages = (
            dataframe.groupby(["Offpersonnelbasic"])["Offpersonnelbasic"]
            .agg(["count"])
            .reset_index()
        )
        dataframe = dataframe[
            ~dataframe["Offpersonnelbasic"].isin(
                personnel_packages[personnel_packages["count"] < 10][
                    "Offpersonnelbasic"
                ]
            )
        ]

    # extra data cleaning pieces to make downs be 1-4, runpass to only be R or P, and offensive personnel to be known and have 11 players on the field
    if target == "Runpass":
        dataframe = dataframe[(~dataframe.Offpersonnelbasic.isin(["Unknown"]))]
    elif target == "Playaction":
        dataframe = dataframe[
            (~dataframe.Offpersonnelbasic.isin(["Unknown"]))
            & (dataframe["Runpass"] == "P")
        ]
    elif target == "Passdirection":
        dataframe = dataframe[
            (dataframe["Offpersonnelbasic"] != "Unknown")
            & (dataframe["Offpersonnelbasic"] != "10 Men")
            & (dataframe["Passdirection"] != "X")
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

    logger.info("starting prediction set process")
    logger.info(f"personnel packages: {personnel}")

    all_combo_set = product(
        quarter,
        minutes_left,
        down,
        distance,
        personnel,
        yards_on_previous_play,
        field_position,
        score_differential,
        previous_play,
    )

    final_test = pd.DataFrame(
        all_combo_set,
        columns=[
            "Quarter",
            "Minutes Left",
            "Down",
            "Distance",
            "Offpersonnelbasic",
            "Yards on Previous Play",
            "Fieldposition",
            "Scoredifferential",
            "previous_play",
        ],
    )

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
    print(output_file_path)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_file_path, index=False)


def final_prediction(model, predictions, predictors, target, le):
    logger.info("starting to predict")
    final_test_dummy = pd.get_dummies(
        predictions, columns=["Offpersonnelbasic", "previous_play"]
    )

    model_predictions = model.predict(final_test_dummy)
    logger.info("Completed prediction process")

    predictions[f"Predicted {target}"] = model_predictions

    return predictions


def randomforest_auto(
    file_path: str, target: str, output: str, team: str, year: str
) -> pd.DataFrame:
    start = time()
    # call cleaner data to clean the dataset for ML use
    dataframe = pd.read_csv(file_path)

    cleaned_data = cleaner(dataframe, target)

    cleaned_data = cleaned_data.astype(
        {"Yards on Previous Play": "int", "Minutes Left": "int"}
    )

    cleaned_data_w_dummies = pd.get_dummies(
        cleaned_data, columns=["previous_play", "Offpersonnelbasic"]
    )
    logger.info("completed data cleaning with dummies")

    # predictor variables used: all variables besides the target variable
    predictors = cleaned_data_w_dummies.columns.drop([target])
    target = target  # target variable

    # splits the subset into a training set to fit the models on and a testing set to
    # test the models on for their accuracy
    train_data, test_data, train_sln, test_sln = train_test_split(
        cleaned_data_w_dummies[predictors],
        cleaned_data_w_dummies[target],
        test_size=0.2,
        random_state=0,
    )

    estimators = np.arange(start=25, stop=151, step=25)

    criterions = ["gini", "entropy"]

    # house all the accuracies when tuning parameters
    results = {
        "max_depth": [],
        "n_estimators": [],
        "min_samples_split": [],
        "min_samples_leaf": [],
        "max_features": [],
        "criterion": [],
    }

    # house the paramters for the model
    params = {
        "max_depth": 0,
        "n_estimators": 0,
        "min_samples_split": 0,
        "min_samples_leaf": 0,
        "max_features": 0,
        "criterion": "",
    }
    max_accuracy = 0

    forest = RandomForestClassifier(random_state=0)
    forest.fit(train_data, train_sln)
    prediction = forest.predict(test_data)
    val = accuracy_score(test_sln, prediction)
    logger.info(f"baseline forest accuracy: {val}")

    # finding the optimal depth for the decision trees to go to
    for i in range(1, 30):
        forest = RandomForestClassifier(max_depth=i, random_state=0)
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = accuracy_score(test_sln, prediction)
        results["max_depth"].append(val)

    max_accuracy = max(results["max_depth"])
    logger.info(f"Best Accuracy after tuning max depth is: {max(results['max_depth'])}")
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
        logger.info("No improvement made. N Estimators given default value")
        max_estimators = 100
        params["n_estimators"] = estimators[max_estimators]
    else:
        logger.info(
            f"Best Accuracy after tuning n_estimators is: {max(results['n_estimators'])}"
        )
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
        logger.info("No improvement made. Min Samples Split given default value")
        max_samp_split = 2
        params["min_samples_split"] = 2
    else:
        logger.info(
            f"Best Accuracy after tuning min samples split is: {max(results['min_samples_split'])}"
        )
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
        logger.info("No improvement made. Min Samples Leaf given default value")
        max_samp_leaf = 1
        params["min_samples_leaf"] = 1
    else:
        logger.info(
            f"Best Accuracy after tuning min samples leaf is: {max(results['min_samples_leaf'])}"
        )
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
        logger.info("No improvement made. Min Samples Leaf given default value")
        max_feat = len(cleaned_data.columns) ** 0.5
        params["max_features"] = max_feat
    else:
        # print max accuracy to see if it improves
        logger.info(
            f"Best Accuracy after tuning max features is: {max(results['max_features'])}"
        )
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

    # print max accuracy to see if it improves
    logger.info(f"Best Accuracy after tuning criterion is: {max(results['criterion'])}")

    max_accuracy = max(results["criterion"])
    max_criterion = find_max(results["criterion"])
    params["criterion"] = criterions[max_criterion]

    logger.info(f"Final Model Accuracy {max_accuracy}")

    logger.info(f"Tuned parameters: {params}")

    logger.info(f"Random Forest Feature Importance {forest.feature_importances_}")

    final_test = prediction_set(cleaned_data)

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
    logger.info("completed predicting")

    final_test[f"Predicted {target}"] = predictions

    grouped = (
        final_test.groupby(
            [
                "Offpersonnelbasic",
                "Down",
                "Distance",
                "Minutes Left",
                f"Predicted {target}",
            ]
        )[f"Predicted {target}"]
        .agg(["count"])
        .reset_index()
    )
    logger.info(grouped.head())

    grouped["percentage"] = grouped["count"] / grouped.groupby(
        ["Offpersonnelbasic", "Down", "Distance", "Minutes Left"]
    )["count"].transform("sum")
    logger.info(grouped.head())

    download_data(grouped, f"{output}/{team}/{target}_{year}.csv")

    logger.info(f"Time Duration: {(time() - start) / 60} minutes")
    return grouped


def main():
    args = Arguments().parse()
    logger.info(args)

    randomforest_auto(args.file, args.predictor, args.output, args.team, args.year)


if __name__ == "__main__":
    main()
