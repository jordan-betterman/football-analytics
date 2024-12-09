import logging as logger

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

logger.getLogger().setLevel(logger.INFO)


def find_max(param):
    max_index = 0
    agg = 0
    for i in param:
        if i > param[max_index]:
            max_index = agg
        agg += 1
    return max_index


def rf_regress_params_tuner(train_data, test_data, train_sln, test_sln, cleaned_data):
    estimators = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

    results = {
        "max_depth": [],
        "n_estimators": [],
        "min_samples_split": [],
        "min_samples_leaf": [],
        "max_features": [],
    }  # house all the accuracies when tuning parameters

    params = {
        "max_depth": 0,
        "n_estimators": 0,
        "min_samples_split": 0,
        "min_samples_leaf": 0,
        "max_features": 0,
    }  # house the paramters for the model

    max_r2 = 0

    # finding the optimal depth for the decision trees to go to
    for i in range(1, 30):
        forest = RandomForestRegressor(max_depth=i, random_state=0)
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = r2_score(test_sln, prediction)
        results["max_depth"].append(val)

    max_r2 = max(results["max_depth"])
    logger.info(f"Best r2 after tuning max depth is: {max(results['max_depth'])}")
    depth = find_max(results["max_depth"]) + 1
    params["max_depth"] = depth

    # find the optimal amount of decision trees to use
    for i in estimators:
        forest = RandomForestRegressor(max_depth=depth, n_estimators=i, random_state=0)
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = r2_score(test_sln, prediction)
        results["n_estimators"].append(val)

    if max(results["n_estimators"]) < max_r2:
        logger.info("No improvement made. N Estimators given default value")
        max_estimators = 100
        params["n_estimators"] = estimators[max_estimators]
    else:
        logger.info(
            f"Best r2 after tuning n_estimators is: {max(results['n_estimators'])}"
        )
        max_r2 = max(results["n_estimators"])
        max_estimators = find_max(results["n_estimators"])
        params["n_estimators"] = estimators[max_estimators]

    # find the optimal amount of sample splits to use
    for i in range(2, 15):
        forest = RandomForestRegressor(
            max_depth=depth,
            n_estimators=estimators[max_estimators],
            min_samples_split=i,
            random_state=0,
        )
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = r2_score(test_sln, prediction)
        results["min_samples_split"].append(val)

    if max(results["min_samples_split"]) < max_r2:
        logger.info("No improvement made. Min Samples Split given default value")
        max_samp_split = 2
        params["min_samples_split"] = 2
    else:
        logger.info(
            f"Best r2 after tuning min samples split is: {max(results['min_samples_split'])}"
        )
        max_r2 = max(results["min_samples_split"])
        max_samp_split = find_max(results["min_samples_split"]) + 2
        params["min_samples_split"] = max_samp_split

    # find the optimal amount of sample leafs to use
    for i in range(1, 15):
        forest = RandomForestRegressor(
            max_depth=depth,
            n_estimators=estimators[max_estimators],
            min_samples_split=max_samp_split,
            min_samples_leaf=i,
            random_state=0,
        )
        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = r2_score(test_sln, prediction)
        results["min_samples_leaf"].append(val)

    if max(results["min_samples_leaf"]) < max(results["min_samples_split"]):
        logger.info("No improvement made. Min Samples Leaf given default value")
        max_samp_leaf = 1
        params["min_samples_leaf"] = 1
    else:
        logger.info(
            f"Best r2 after tuning min samples leaf is: {max(results['min_samples_leaf'])}"
        )
        max_r2 = max(results["min_samples_leaf"])
        max_samp_leaf = find_max(results["min_samples_leaf"]) + 1
        params["min_samples_leaf"] = max_samp_leaf

    # find the optimal amount of features to use
    for i in range(1, len(cleaned_data.columns)):
        forest = RandomForestRegressor(
            max_depth=depth,
            n_estimators=estimators[max_estimators],
            min_samples_split=max_samp_split,
            min_samples_leaf=max_samp_leaf,
            max_features=i,
            random_state=0,
        )

        forest.fit(train_data, train_sln)
        prediction = forest.predict(test_data)
        val = r2_score(test_sln, prediction)
        results["max_features"].append(val)

    if max(results["max_features"]) < max_r2:
        logger.info("No improvement made. Min Samples Leaf given default value")
        max_feat = len(cleaned_data.columns) ** 0.5
        params["max_features"] = max_feat
    else:
        logger.info(
            f"Best r2 after tuning max features is: {max(results['max_features'])}"
        )  # print max accuracy to see if it improves
        max_r2 = max(results["max_features"])
        max_feat = find_max(results["max_features"]) + 1
        params["max_features"] = max_feat

    logger.info(f"Final Model Accuracy {max_r2}")

    logger.info(f"Tuned parameters: {params}")

    logger.info(f"Random Forest Feature Importance {forest.feature_importances_}")
    return params
