# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from datetime import datetime
import logging

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

from sklearn.utils import shuffle
from arguments import Arguments
from pathlib import Path

pd.options.mode.chained_assignment = None

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def cleaner(dataframe, target):
    dataframe["Gainloss"] = dataframe["Gainloss"].replace(" ", np.nan, regex=True)

    # created a new column that moved the yards gained or lost on a play down a row
    dataframe["Yards on Previous Play"] = dataframe["Gainloss"].shift()
    dataframe["previous_play"] = dataframe["Runpass"].shift()

    dataframe = dataframe[(dataframe["Runpass"] == "R") | (dataframe["Runpass"] == "P")]
    dataframe = dataframe[
        (dataframe["previous_play"] == "R") | (dataframe["previous_play"] == "P")
    ]

    # extra data cleaning pieces to make downs be 1-4,
    dataframe = dataframe[dataframe["Down"] > 0]

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

    # runpass to only be R or P, and offensive personnel to be known and have 11 players on the field
    if target == "Runpass":
        dataframe = dataframe[
            (~dataframe.Offpersonnelbasic.isin(["Unknown", "10 Men"]))
        ]
    elif target == "Playaction":
        dataframe = dataframe[
            (~dataframe.Offpersonnelbasic.isin(["Unknown", "10 Men"]))
            & (dataframe["Runpass"] == "P")
        ]

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
            "Hash",
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
            "Yards on Previous Play",
            "Scoredifferential",
            "previous_play",
            "Hash",
        ]
    )

    return subset


def prediction_set(dataset):
    quarter = dataset["Quarter"].unique().tolist()
    down = dataset["Down"].unique().tolist()
    distance = range(1, 11)
    minutes_left = range(1, 16)
    personnel = dataset["Offpersonnelbasic"].unique().tolist()
    previous_play = ["R", "P"]
    score_differential = range(-14, 14, 7)
    yards_on_previous_play = [-10, -5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    field_position = [-30, -35, -40, -45, 50, 45, 40, 35, 30]
    hash = ["L", "C", "R"]

    logging.info("starting prediction set process")
    logging.info(f"personnel packages: {personnel}")

    all_combo_set = product(
        quarter,
        minutes_left,
        down,
        distance,
        field_position,
        personnel,
        yards_on_previous_play,
        score_differential,
        previous_play,
        hash,
    )
    logging.info("cross product completed")

    final_test = pd.DataFrame(
        all_combo_set,
        columns=[
            "Quarter",
            "Minutes Left",
            "Down",
            "Distance",
            "Fieldposition",
            "Offpersonnelbasic",
            "Yards on Previous Play",
            "Scoredifferential",
            "previous_play",
            "Hash",
        ],
    )
    logging.info("completed prediction set")

    return final_test


def download_data(dataset, name):
    output_file_path = Path(name)
    logging.info(output_file_path)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_file_path, index=False)


def final_prediction(model, predictions, predictors, target, le):
    logging.info("starting to predict")
    final_test_dummy = pd.get_dummies(
        predictions, columns=["Offpersonnelbasic", "previous_play"]
    )

    model_predictions = model.predict(final_test_dummy)
    logging.info("Completed prediction process")

    predictions[f"Predicted {target}"] = model_predictions

    return predictions


def randomforest_auto(
    file_path: str, target: str, output: str, team: str, year: str
) -> pd.DataFrame:
    start = datetime.now().replace(microsecond=0)
    # call cleaner data to clean the dataset for ML use
    dataframe = pd.read_csv(file_path)

    cleaned_data = cleaner(dataframe, target)

    cleaned_data_w_dummies = cleaned_data.astype(
        {
            "Yards on Previous Play": "int32",
            "Minutes Left": "int32",
            "Down": "int32",
            "Quarter": "int32",
            "Distance": "int32",
            "Scoredifferential": "int32",
            "Fieldposition": "int32",
            "previous_play": "category",
            "Offpersonnelbasic": "category",
        }
    )
    cleaned_data_w_dummies = cleaned_data_w_dummies.drop_duplicates()

    cleaned_data_w_dummies = shuffle(cleaned_data_w_dummies, random_state=0)

    cleaned_data_w_dummies = pd.get_dummies(
        cleaned_data_w_dummies,
        columns=["Offpersonnelbasic", "previous_play", "Hash"],
        dtype=np.int8,
    )
    logging.info("completed data cleaning with dummies")

    # predictor variables used: all variables besides the target variable
    predictors = cleaned_data_w_dummies.columns.drop([target])

    # splits the subset into a training set to fit the models on and a testing set to
    # test the models on for their accuracy
    x_train, x_test, y_train, y_test = train_test_split(
        cleaned_data_w_dummies[predictors],
        cleaned_data_w_dummies[target],
        test_size=0.2,
        random_state=0,
    )

    ros = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(x_train, y_train)

    logging.info(f"Playcall distribution after oversampling: {Counter(y_resampled)}")

    baselineforest = RandomForestClassifier(
        random_state=0, n_estimators=100, max_features="sqrt", oob_score=True, verbose=1
    )
    baselineforest.fit(x_resampled, y_resampled)
    predicted = baselineforest.predict(x_test)
    prediction_prob = baselineforest.predict_proba(x_test)[:, -1]
    baseline_auc = roc_auc_score(y_test, prediction_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

    logging.info("---" * 20)
    logging.info("Baseline Random Forest Evaluation Scores")
    logging.info(f"OOB Score: {round(baselineforest.oob_score_, 4) * 100}%")
    logging.info(f"AUC: {baseline_auc}")
    logging.info(f"Specificity: {round(tn / (tn + fp), 4) * 100}%")
    logging.info(f"Sensitivity: {round(round(tp / (tp + fn), 4) * 100, 2)}%")

    logging.info(
        f"Test Set Accuracy: {round(accuracy_score(y_test, baselineforest.predict(x_test)), 4) * 100}%"
    )

    m = []
    oob_error_rate = []

    for i in range(2, len(cleaned_data_w_dummies.columns)):
        forest = RandomForestClassifier(
            random_state=0, n_estimators=100, max_features=i, oob_score=True
        )
        forest.fit(x_resampled, y_resampled)

        m.append(i)
        oob_error_rate.append(round(1 - forest.oob_score_, 4))

    logging.info(
        f"Best max feature value: {oob_error_rate.index(min(oob_error_rate)) + 2}"
    )

    tuned_forest = RandomForestClassifier(
        random_state=0,
        n_estimators=100,
        max_features=(oob_error_rate.index(min(oob_error_rate)) + 2),
        oob_score=True,
        verbose=1,
    )
    tuned_forest.fit(x_resampled, y_resampled)

    prediction_prob = tuned_forest.predict_proba(x_test)[:, -1]
    tuned_predicted = tuned_forest.predict(x_test)
    auc = roc_auc_score(y_test, prediction_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, tuned_predicted).ravel()

    logging.info("---" * 20)
    logging.info("Tuned Random Forest Evaluation Scores")
    logging.info(f"OOB Score: {tuned_forest.oob_score_}")
    logging.info(f"AUC: {auc}")
    logging.info(f"Specificity: {round((tn / (tn + fp)), 4) * 100}%")
    logging.info(f"Sensitivity: {round(round((tp / (tp + fn)), 4) * 100, 2)}%")

    logging.info(
        f"Test Set Accuracy: {round(accuracy_score(y_test, tuned_forest.predict(x_test)), 4) * 100}%"
    )

    final_test = prediction_set(cleaned_data)

    final_x_test = pd.get_dummies(
        final_test, columns=["Offpersonnelbasic", "previous_play", "Hash"]
    )

    if tuned_forest.oob_score_ > baselineforest.oob_score_:
        logging.info("Using Tuned Forest For Final Predictions")
        final_test_predictions = tuned_forest.predict(final_x_test)
    else:
        logging.info("Using BaselineForest For Final Predictions")
        final_test_predictions = baselineforest.predict(final_x_test)

    logging.info("Predictions Complete")

    final_test[f"Predicted {target}"] = final_test_predictions

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

    grouped["percentage"] = grouped["count"] / grouped.groupby(
        ["Offpersonnelbasic", "Down", "Distance", "Minutes Left"]
    )["count"].transform("sum")

    download_data(grouped, f"{output}/{team}/{target}_{year}.csv")

    logging.info(
        f"Time Duration: {(datetime.now().replace(microsecond=0) - start) / 60} minutes"
    )
    return grouped


def main():
    args = Arguments().parse()

    randomforest_auto(args.file, args.predictor, args.output, args.team, args.year)


if __name__ == "__main__":
    main()
