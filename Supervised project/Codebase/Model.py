import tensorflow as tf
from tensorflow.keras import optimizers
from SupervisedSolver import SupervisedSolver
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import numpy as np


class Model(SupervisedSolver):
    def __init__(self, method, nrFeatures, epochs=None, batchSize=None, depth=None):
        methods = {
            "neuralNetwork": [self.NN, "tf"],
            "convNeuralNetwork": [self.conv_NN, "tf"],
            "GRU_NN": [self.GRU_NN, "tf"],
            "decisionTree": [self.decision_tree, "sklearn"],
            "xGBoost": [self.xGBoost, "sklearn"],
        }
        self.nrFeatures = nrFeatures
        self.initMethod, self.tool = methods[method]
        self.epochs = epochs
        self.batchSize = batchSize
        self.depth = depth
        self.initMethod()

    def __call__(self):
        return self.model

    def NN(self):
        """
        Initializes the model, setting up layers and compiles the model,
        prepping for training.

        Returns:
            tensorflow_object: compiled model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    500,
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                    input_shape=(self.nrFeatures,),
                ),
                tf.keras.layers.Dense(800, activation="tanh"),
                tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1300, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
                
                tf.keras.layers.Dropout(0.5),   
                tf.keras.layers.Dense(1100, activation="tanh"),
                tf.keras.layers.Dense(600, activation="tanh"),
                tf.keras.layers.Dropout(0.5), 
                tf.keras.layers.Dense(200, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
                tf.keras.layers.Dense(50, activation="tanh"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        
        self.optimizer = optimizers.Adam()
        model.compile(
            loss="binary_crossentropy", optimizer=self.optimizer, metrics=["accuracy"]
        )
        self.fit = lambda X, y: self.model.fit(
            X, y, epochs=self.epochs, batch_size=self.batchSize
        )
        self.predict = lambda X: np.around(model(X).numpy().ravel())
        self.model = model
        self.model.summary()

    def conv_NN(self):
        """
        Initializes the model, setting up layers and compiles the model,
        prepping for training.

        Returns:
            tensorflow_object: compiled model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    64, (3), input_shape=(self.nrFeatures, 1), activation="relu"
                ),
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation="relu"),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(),
            metrics=["accuracy"],
        )

        self.fit = lambda X, y: self.model.fit(
            X.reshape(X.shape[0], X.shape[1], 1),
            y.reshape(y.shape[0], 1, 1),
            epochs=self.epochs,
            batch_size=self.batchSize,
        )
        self.predict = lambda X: np.around(
            self.model(X.reshape(X.shape[0], X.shape[1], 1)).numpy().ravel()
        )
        self.model = model

    def GRU_NN(self):
        """
        Initializes the model, setting up layers and compiles the model,
        prepping for training.

        Returns:
            tensorflow_object: compiled model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(64, return_sequences=True),
                    input_shape=(self.nrFeatures, 1),
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation="relu"),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        optimizer = optimizers.Adam()
        model.compile(
            loss="binary_crossentropy",
            metrics=["accuracy"],
            optimizer=optimizer,
        )

        self.fit = lambda X, y: self.model.fit(
            X.reshape(X.shape[0], X.shape[1], 1),
            y.reshape(y.shape[0], 1, 1),
            epochs=self.epochs,
            batch_size=self.batchSize,
        )
        self.predict = lambda X: np.around(
            model(X.reshape(X.shape[0], X.shape[1], 1)).numpy().ravel()
        )
        self.model = model

    def decision_tree(self):
        # Prefers max_depth = 12
        model = DecisionTreeRegressor(max_depth=self.depth)
        self.fit = lambda X, y: self.model.fit(X, y)
        self.predict = lambda X: np.around(self.model.predict(X).ravel())
        self.model = model

    def xGBoost(self):
        # Prefers max_depth = 5
        self.fit = lambda X, y: self.model.fit(X, y)
        self.model = xgb.XGBClassifier(
            max_depth=self.depth,
            use_label_encoder=False,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            eta=0.1,
        )
        self.predict = lambda X: np.around(self.model.predict(X).ravel())
