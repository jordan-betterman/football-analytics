# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# import logging as logger

# logger.getLogger().setLevel(logger.INFO)



# le = LabelEncoder()
#     le.fit(train_sln)
#     labeled = le.transform(train_sln)
#     labeled_test = le.transform(test_sln)

#     #setting grid of selected parameters for iteration
#     params = {'gamma': [0,0.1,0.2,0.4],
#                 'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3],
#                 'max_depth': np.arange(start=1, stop=30, step=1, dtype='int16'),
#                 'n_estimators': np.arange(start=50, stop=150, step=25, dtype='int16'),
#                 'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7],
#                 'min_child_weight': np.arange(start=1, stop=7, step=1, dtype='int16'),
#                 'subsample': np.arange(start=0.4, stop=1, step=0.1)}
#     xgbc = xgb.XGBRFClassifier(objective='binary:logistic', booster="gbtree", eval_metric='auc', tree_method='hist', grow_policy='lossguide', random_state=0, verbosity=0)
#     logger.info("starting bayes search on xgboost")
#     clf = BayesSearchCV(estimator=xgbc, search_spaces=params, scoring='roc_auc', return_train_score=True, cv=5, n_jobs=-1, n_iter=10, random_state=0)
#     xgboost = clf.fit(train_data, labeled.ravel())

#     train_predictions = clf.predict(train_data)
#     test_predictions = clf.predict(test_data)

#     train_accuracy = accuracy_score(train_sln, le.inverse_transform(train_predictions))
#     test_accuracy = accuracy_score(test_sln, le.inverse_transform(test_predictions))


#     logger.info(f"train accuracy {train_accuracy}")
#     logger.info(f"test accuracy {test_accuracy}")
#     logger.info(f"best accuracy score: {xgboost.best_score_}")
#     logger.info(f"test accuracy score: {xgboost.score(test_data, labeled_test)}")
#     logger.info(f"parameters: {xgboost.best_estimator_.get_params()}" )

#     forest_params = {
#     'max_depth': np.arange(start=1, stop=30, step=1, dtype='int16'),
#     'n_estimators': np.arange(start=50, stop=150, step=25, dtype='int16'),
#     'max_features': list(range(1, len(cleaned_data_w_dummies.columns))),
#     'min_samples_split': list(range(2,15)),
#     'min_samples_leaf': list(range(1,15))
#     }
#     logger.info("starting bayes search on random forest")
#     forest = BayesSearchCV(
#         estimator=RandomForestClassifier(random_state=0),
#         search_spaces=forest_params,
#         n_iter=10,
#         cv=5,
#         return_train_score=True,
#         scoring="roc_auc",
#         random_state=0
#     )

#     forest.fit(train_data, train_sln)

#     logger.info(f"best accuracy score: {forest.best_score_}")
#     logger.info(f"test accuracy score: {forest.score(test_data, test_sln)}")
#     logger.info(f"parameters: {forest.best_estimator_.get_params()}" )

#     predictions = pd.read_csv(f"down_breaks/prediction_sets/{team}_{year}_prediction_set.csv")

#     logger.info("Prediction set completed")

#     # evaluate if the xgboost or random forest model is better fit for the data
#     if xgboost.best_score_ > forest.best_score_:
#         logger.info("xgboost best score > forest best score... using xgboost model")
#         del forest
#         final_test = final_prediction(xgboost,predictions,predictors,target,le)
#         final_test[f"Predicted {target}"] = le.inverse_transform(final_test[f"Predicted {target}"])
#     elif xgboost.best_score_ < forest.best_score_:
#         logger.info("xgboost best score < forest best score... using forest model")
#         del xgboost
#         final_test = final_prediction(forest,predictions,predictors,target,le)
#         final_test[f"Predicted {target}"] = le.inverse_transform(final_test[f"Predicted {target}"])
#     else:
#         logger.info("xgboost best score == forest best score... using xgboost model")
#         del forest
#         final_test = final_prediction(xgboost,predictions,predictors,target,le) #xgboost is faster so it will win the tiebreaker
#         final_test[f"Predicted {target}"] = le.inverse_transform(final_test[f"Predicted {target}"])