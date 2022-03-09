import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
from sklearn.model_selection import train_test_split



def create_model(features):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                30,
                activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                input_shape=(features,),
            ),
            tf.keras.layers.Dense(
                15,
                activation="relu",
            ),
            
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = optimizers.Adam()

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    return model


if __name__ == "__main__":
    
    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()
    
    X_data = data.data
    y_data = data.target

    features = np.shape(X_data)[1]
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
    
    
    model = create_model(features)

    with tf.device("/CPU:0"):
        
        model.fit(X_train, y_train, epochs=20)
        y_tilde = np.around(model.predict(X_test))

   
    acc = np.sum(np.equal(y_tilde.ravel(), y_test)) / len(y_test) * 100

    print(f"Accuracy: {acc:.1f}%")
