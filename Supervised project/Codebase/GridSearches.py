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

def gridNN():
    DH.fillWithImputer()
    DH.standardScale()
    X,Y = DH(include_test=False)
    
    tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='GridSearches',
                     project_name='NN',
                     overwrite = True)

    tuner.search(X, Y, epochs=100, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number nodes in first and second layer is {best_hps.get('num_of_neurons1')} and \
    {best_hps.get('num_of_neurons2')} the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)
    
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
                    50,
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                    input_shape=(30,),
                ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(
                    units=hp.Int('num_of_neurons1', min_value=10, max_value=50, step=5),
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(
                    units=hp.Int('num_of_neurons2', min_value=10, max_value=50, step=5), 
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(units=10, activation="tanh"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3]) 
    optimizer = optimizers.Adam(learning_rate=hp_learning_rate)
    model.compile(
        loss="binary_crossentropy", optimizer= optimizer, metrics=["accuracy"]
    )
    return model


gridNN()

#gridXGBoost()


