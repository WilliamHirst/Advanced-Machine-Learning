import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import PredefinedSplit
from xgboost import XGBClassifier
from DataHandler import DataHandler
from tensorflow.keras import optimizers
import keras_tuner as kt
from Functions import timer

# for custom activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})

def decisionTrees():
    import joblib
    from joblib import dump, load
    import os

    start_time = timer(None)
    DH.setNanToMean()
    DH.split()
    X_train, X_val, y_train, y_val = DH(include_test=True)
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    model = DecisionTreeRegressor(max_features = "auto")

    params = {
        "max_depth": [7,9,11,12],
        "min_samples_leaf":[6,7,8],
    }
    """params={"splitter":["best","random"],
            "max_depth" : [7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }"""

    pds = PredefinedSplit(test_fold=split_index)

    GridSearch = GridSearchCV(
        model, param_grid=params, scoring="roc_auc", cv=pds, verbose=3
    )
    GridSearch.fit(X, y)
    timer(start_time)
    best_model = GridSearch.best_estimator_
    best_score = np.sum(np.where(np.around(best_model.predict(X_val)) == y_val, 1, 0)) / len(X_val)

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
    

def gridXGBoost():
    import joblib
    from joblib import dump, load
    import os

    start_time = timer(None)
    DH.split()
    X_train, X_val, y_train, y_val = DH(include_test=True)
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    xgb = XGBClassifier(
        use_label_encoder=False,
        objective="binary:logistic",
        eval_metric="error",
        tree_method="hist",
        max_features=20,
        nthread=1,
        subsample=0.9,
        gamma=0.1,
        verbosity=0,
    )
    params = {
        "n_estimator": [50, 100, 200, 300, 400, 500],
        "max_dept": [1, 2, 3, 4, 5, 6],
        "eta": [1e-1, 5e-2, 1e-2, 5e-3],
    }

    pds = PredefinedSplit(test_fold=split_index)

    GridSearch = GridSearchCV(
        xgb, param_grid=params, scoring="roc_auc", cv=pds, verbose=3
    )
    GridSearch.fit(X, y)
    timer(start_time)
    best_model = GridSearch.best_estimator_
    best_score = np.sum(np.where(best_model.predict(X_val) == y_val, 1, 0)) / len(X_val)

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


def gridSVM():
    import joblib
    from joblib import dump, load
    import os
    from sklearn.svm import OneClassSVM
    from sklearn import linear_model

    DH.fillWithImputer()
    DH.standardScale()

    start_time = timer(None)
    X_b, y_b, X_all, y_all = DH.AE_prep()

    params = {
        "kernel": ["poly", "rbf"],
        "nu": [0.01, 0.05, 0.1, 0.2, 0.3],
        "max_iter": [1000, 10000, 50000],
    }
    best_kernel = "linear"
    best_nu = 0.01
    best_max_iter = 10
    best_score = 0
    for kernel in params["kernel"]:
        for nu in params["nu"]:
            for max_iter in params["max_iter"]:
                print(f"Kernel: {kernel}, Nu: {nu}, Max_iter: {max_iter}")
                svm = OneClassSVM(kernel=kernel, nu=nu, max_iter=max_iter)
                svm.fit(X_b)
                prediction = np.around(svm.predict(X_all))
                predictin = np.where(prediction == -1, 1, 0)
                score = np.sum(np.equal(prediction, y_all)) / len(y_all) * 100
                print(f"{score} %")
                if score > best_score:
                    best_score = score
                    best_kernel = kernel
                    best_nu = nu
                    best_max_iter = max_iter

    timer(start_time)

    print("\n Best score:")
    print(best_score)

    print("\n Best hyperparameters: ")
    print(f"Kernel: {kernel}, Nu: {nu}, Max_iter: {max_iter}")


def gridautoencoder():
    DH.setNanToMean()#DH.fillWithImputer()
    DH.standardScale()
    X_b, y_b, X_all, y_all, X_back_test, X_sig_test = DH.AE_prep(whole_split=True)

    start_time = timer(None)
    tuner = kt.Hyperband(
        AE_model_builder,
        objective=kt.Objective("val_mse", direction="min"),
        max_epochs=50,
        factor=3,
        directory="GridSearches",
        project_name="AE",
        overwrite=True,
    )

    tuner.search(X_b, X_b, epochs=100, validation_data=(X_back_test, X_back_test))
    timer(start_time)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    For Encoder: \n 
    First layer has {best_hps.get('num_of_neurons0')} with activation {best_hps.get('0_act')} \n
    Second layer has {best_hps.get('num_of_neurons1')} with activation {best_hps.get('1_act')} \n
    
    Latent layer has {best_hps.get("lat_num")} with activation {best_hps.get('2_act')} \n
    \n
    For Decoder: \n 
    First layer has {best_hps.get('num_of_neurons5')} with activation {best_hps.get('5_act')}\n
    Second layer has {best_hps.get('num_of_neurons6')} with activation {best_hps.get('6_act')}\n
    Third layer has activation {best_hps.get('7_act')}\n
    \n
    with learning rate = {best_hps.get('learning_rate')}
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


def AE_model_builder(hp):
    inputs = tf.keras.layers.Input(shape=30, name="encoder_input")
    x = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons0", min_value=17, max_value=30, step=1),
        activation=hp.Choice("0_act", ["relu", "tanh", "leakyrelu"]),
    )(inputs)
    x1 = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons1", min_value=9, max_value=16, step=1),
        activation=hp.Choice("1_act", ["relu", "tanh", "leakyrelu"]),
    )(x)
    val = hp.Int("lat_num", min_value=1, max_value=8, step=1)
    x2 = tf.keras.layers.Dense(
        units=val, activation=hp.Choice("2_act", ["relu", "tanh", "leakyrelu"])
    )(x1)
    encoder = tf.keras.Model(inputs, x2, name="encoder")

    latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")
    x = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons5", min_value=9, max_value=16, step=1),
        activation=hp.Choice("5_act", ["relu", "tanh", "leakyrelu"]),
    )(latent_input)
    x1 = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons6", min_value=17, max_value=30, step=1),
        activation=hp.Choice("6_act", ["relu", "tanh", "leakyrelu"]),
    )(x)
    output = tf.keras.layers.Dense(
        30, activation=hp.Choice("7_act", ["relu", "tanh", "leakyrelu", "sigmoid"])
    )(x1)
    decoder = tf.keras.Model(latent_input, output, name="decoder")

    outputs = decoder(encoder(inputs))
    AE_model = tf.keras.Model(inputs, outputs, name="AE_model")

    hp_learning_rate = hp.Choice("learning_rate", values=[9e-2, 9.5e-2, 1e-3, 1.5e-3])
    optimizer = optimizers.Adam(hp_learning_rate)
    AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

    return AE_model


if __name__ == "__main__":
    tf.random.set_seed(1)
    DH = DataHandler("rawFeatures_TR.npy", "rawTargets_TR.npy")

    #with tf.device("/CPU:0"):
        # gridNN()
    #    gridautoencoder()

    # gridXGBoost()
    # gridSVM()
    decisionTrees()
