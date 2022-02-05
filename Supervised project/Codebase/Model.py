import tensorflow as tf
from tensorflow.keras import optimizers
from SupervisedSolver import SupervisedSolver
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb





class Model(SupervisedSolver):
    def __init__(self, method, nrFeatures, epochs, batchSize, depth):
        methods = { "neuralNetwork": [self.neural_network, "tf"],
                    "convNeuralNetwork": [self.convolutional_neural_network, "tf"],
                    "decisionTree": [self.decision_tree, "sklearn"],
                    "xGBoost": [self.xGBoost, "sklearn"]}
        self.nrFeatures  = nrFeatures
        self.initMethod, self.tool  = methods[method]
        self.epochs = epochs
        self.batchSize = batchSize
        self.depth = depth
        self.initMethod()
    
    def __call__(self):
        return self.model
    
    def neural_network(self):
        """
        Initializes the model, setting up layers and compiles the model,
        prepping for training.

        Returns:
            tensorflow_object: compiled model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    100, activation="relu", input_shape=(self.nrFeatures,)
                ),
                tf.keras.layers.Dense(60, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=1e-4)
        model.compile(loss="binary_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])
        self.fit = lambda X, y: self.model.fit(X, y, epochs = self.epochs, batch_size = self.batchSize)
        self.model = model
    def convolutional_neural_network(self):
        """
        Initializes the model, setting up layers and compiles the model,
        prepping for training.

        Returns:
            tensorflow_object: compiled model
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(64, (3), input_shape=(self.nrFeatures,1), activation='relu'),
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100, activation="relu"),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(20, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.optimizer = optimizers.Adam()
        model.compile(loss= "binary_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])
        self.fit = lambda X, y: self.model.fit(X.reshape(X.shape[0], X.shape[1], 1),
                                               y.reshape(y.shape[0], 1, 1), 
                                               epochs = self.epochs, 
                                               batch_size = self.batchSize)
        self.model = model


    def decision_tree(self):
        model = DecisionTreeRegressor()
        self.fit = lambda X, y: self.model.fit(X, y)
        self.model = model
    def xGBoost(self):
        self.fit = lambda X, y: self.model.fit(X, y)
        
        self.model = xgb.XGBClassifier(max_depth=self.depth,
                                       use_label_encoder=False,
                                       objective = "binary:logistic",
                                       eval_metric = "logloss")
        

