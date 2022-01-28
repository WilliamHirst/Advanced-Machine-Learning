import tensorflow as tf
from tensorflow.keras import optimizers
from SupervisedSolver import SupervisedSolver
from sklearn.tree import DecisionTreeRegressor



class Model(SupervisedSolver):
    def __init__(self, method, nrFeatures, ):
        methods = { "neuralNetwork": [self.neural_network, "tf"],
                    "decisionTree": [self.decision_tree, "sklearn"]}
        self.nrFeatures  = nrFeatures
        self.initMethod, self.tool  = methods[method]
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
                    50, activation="sigmoid", input_shape=(self.nrFeatures,)
                ),
                tf.keras.layers.Dense(50, activation="sigmoid"),
                tf.keras.layers.Dense(50, activation="sigmoid"),
                tf.keras.layers.Dense(1),
            ]
        )
        self.optimizer = optimizers.Adam(learning_rate=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])
        self.model = model

    def decision_tree(self):
        model = DecisionTreeRegressor()
        self.model = model
