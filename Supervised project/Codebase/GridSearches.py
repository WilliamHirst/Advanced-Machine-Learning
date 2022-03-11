import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from DataHandler import DataHandler
from tensorflow.keras import optimizers
import keras_tuner as kt
from sklearn.model_selection import PredefinedSplit
from Functions import timer



weights = np.load("../Data/weights.npy")
DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")



def gridXGBoost():
    import joblib
    from joblib import dump, load
    import os
    start_time = timer(None)

    X, Y = DH(include_test=False)
    sum_wpos = sum(weights[i] for i in range(len(X)) if Y[i] == 1.0)
    sum_wneg = sum(weights[i] for i in range(len(X)) if Y[i] == 0.0)

    DH.split()
    X_train, X_val, y_train, y_val = DH(include_test=True)
    split_index = [-1]*len(X_train) + [0]*len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)    
    #scale_pos_weight =  sum_wneg/sum_wpos ,
    xgb = XGBClassifier(
        use_label_encoder=False,
        
        objective="binary:logistic",
        eval_metric= 'ams@0.15',
        tree_method="hist",
        
        max_features=20,
        nthread=1,
        subsample=0.9,
        gamma=0.1,
        verbosity=0,
    )
    params = {
                "n_estimator": [50,100,200,300,400,500],
                "max_dept": [1,2,3,4,5,6],
                "eta": [1e-1,5e-2,1e-2,5e-3]
    } 
                
    pds = PredefinedSplit(test_fold = split_index)

    GridSearch = GridSearchCV(xgb,
                              param_grid=params,
                              scoring="roc_auc",
                              cv=pds,
                              verbose=3)              
    GridSearch.fit(X, y)
    timer(start_time)
    best_model = GridSearch.best_estimator_
    best_score = np.sum(np.where(best_model.predict(X_val) == y_val, 1, 0))/len(X_val)
    
    print("\n Best score:")
    print(best_score)

    print("\n Best hyperparameters:")
    for param, value in best_model.get_params(deep=True).items():
        print(f"{param} -> {value}")

    state = True
    while state == True:
        answ = input("Do you want to save model? (y/n) ")
        if answ == "y":
            name = input("name: ")
            dirname = os.getcwd()
            filename = os.path.join(dirname, f"sklearn_models/model_{name}.joblib")
            print(filename)
            dump(best_model, filename) 
            state = False
            print("Model saved")
        elif answ == "n":
            state = False
            print("Model not saved")


def gridNN():
    DH.fillWithImputer()
    DH.standardScale()
    X, Y = DH(include_test=False)
    start_time = timer(None)
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="GridSearches",
        project_name="NN",
        overwrite=True,
    )

    tuner.search(X, Y, epochs=100, validation_split=0.2)
    timer(start_time)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete. The optimal number nodes in start, first, second layer is {best_hps.get('num_of_neurons0')}, {best_hps.get('num_of_neurons1')} and \
    {best_hps.get('num_of_neurons2')} and third layer {best_hps.get('num_of_neurons3')} the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """
    )

    state = True
    while state == True:
        answ = input("Do you want to save model? (y/n) ")
        if answ == "y":
            name = input("name: ")
            tuner.hypermodel.build(best_hps).save(f"../tf_models/model_{name}.h5")
            state = False
            print("Model saved")
        elif answ == "n":
            state = False
            print("Model not saved")


def model_builder(hp):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons0", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                input_shape=(30,),
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons1", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons2", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(
                units=hp.Int("num_of_neurons3", min_value=10, max_value=50, step=5),
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    hp_learning_rate = hp.Choice("learning_rate", values=[9e-2, 9.5e-2, 1e-3, 1.5e-3])
    optimizer = optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


"""with tf.device("/CPU:0"):
    gridNN()"""

gridXGBoost()
