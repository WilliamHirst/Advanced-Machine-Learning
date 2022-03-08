import tensorflow as tf
from tensorflow.keras import optimizers
from SupervisedSolver import SupervisedSolver
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import numpy as np
from tensorflow.keras.models import Model


class Model(SupervisedSolver):
    def __init__(self, method, nrFeatures, epochs=None, batchSize=None, depth=None):
        methods = {
            "neuralNetwork": [self.NN, "tf"],
            "convNeuralNetwork": [self.conv_NN, "tf"],
            "GRU_NN": [self.GRU_NN, "tf"],
            "decisionTree": [self.decision_tree, "sklearn"],
            "xGBoost": [self.xGBoost, "sklearn"],
            "autoencoder": [self.autoEncoders, "tf"],
            "svm": [self.supportVectorMachines, "tf"],
        }
        self.nrFeatures = nrFeatures
        self.initMethod, self.tool = methods[method]
        self.epochs = epochs
        self.batchSize = batchSize
        self.depth = depth

        self.callback = tf.keras.callbacks.TensorBoard(
            log_dir="../Logs", update_freq="epoch", histogram_freq=10
        )

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
                    50,
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                    input_shape=(self.nrFeatures,),
                ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(
                    30, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(
                    30, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
                ),
                tf.keras.layers.Dense(20, activation="tanh"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.5e-2, decay_steps=10000, decay_rate=0.9
        )
        self.optimizer = optimizers.Adam(learning_rate=lr_schedule)
        model.compile(
            loss="binary_crossentropy", optimizer=self.optimizer, metrics=["accuracy"]
        )

        self.fit = lambda X_train, y_train, X_val, y_val: self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batchSize,
            callbacks=[self.callback],
            validation_split=0.2
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

        self.fit = lambda X_train, y_train, X_val, y_val: self.model.fit(
            X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
            y_train.reshape(y_train.shape[0], 1, 1),
            validation_data=(
                X_val.reshape(X_val.shape[0], X_val.shape[1], 1),
                y_val.reshape(y_val.shape[0], 1, 1),
            ),
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

        self.fit = lambda X_train, y_train, X_val, y_val: self.model.fit(
            X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
            y_train.reshape(y_train.shape[0], 1, 1),
            validation_data=(X_val, y_val),
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
        #Prefers max_depth = 6
        """
        use_label_encoder=False,
        objective = "binary:logistic",
        n_estimators=250,
        eval_metric = "error",
        tree_method = "hist",
        max_features = 10,
        eta = 0.1,
        nthread=1,
        subsample = 0.7
        """                               
        self.model = xgb.XGBClassifier(max_depth=self.depth,
                                    use_label_encoder=False,
                                    objective = "binary:logistic",
                                    n_estimators=500,
                                    eval_metric = "error",
                                    tree_method = "hist",
                                    max_features = 15,
                                    eta = 0.1,
                                    nthread=1,
                                    subsample = 0.9,
                                    gamma = 0.1,
                                    )                                 
        self.fit = lambda X_train, y_train, X_val, y_val: self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)]) 
        self.predict = lambda X: np.around(self.model.predict(X).ravel())

    def supportVectorMachines(self):
        from sklearn import svm

        classifier = svm.SVC(kernel="rbf",)
        self.fit = lambda X_train, y_train: self.model.fit(X_train, y_train)

        self.predict = lambda X: self.model.predict(X)

        self.model = classifier

    def autoEncoders(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    32, activation="relu", input_shape=(self.nrFeatures,)
                ),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(140, activation="sigmoid"),
            ]
        )

        self.optimizer = optimizers.Adam()
        model.compile(loss ="mae",optimizer=self.optimizer, metrics=["accuracy"])

        self.fit = lambda X_train: self.model.fit(
            X_train,
            X_train,
            epochs=20,
            batch_size=512,
            shuffle=True,
        )  # validation_data=(X_all, X_all),

        self.predict = lambda X: np.around(model(X).numpy().ravel())
        self.model = model
