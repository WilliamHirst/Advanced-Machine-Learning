import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from DataHandler import DataHandler




DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")
X,Y = DH(include_test=False)


def gridXGBoost(folds = 5, param_comb = 5):
    xgb = XGBClassifier(
                        max_depth=6,
                        use_label_encoder=False,
                        objective = "binary:logistic",
                        n_estimators=400,
                        eval_metric = "error",
                        tree_method = "hist",
                        max_features = 20,
                        eta = 0.1,
                        nthread=1,
                        subsample = 0.9,
                        gamma = 0.1,
                        verbosity = 0
                            )    
    params = {
        'min_child_weight': [1,2,3,4,5,6,7,8,9,10],
       
        }  

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )

    random_search.fit(X, Y)

    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)


gridXGBoost()


