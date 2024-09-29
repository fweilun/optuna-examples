"""
Optuna example that optimizes a neural network classifier for the
MNIST dataset using Keras and records hyperparameters and metrics using Comet.

In this example, we optimize the number of layers, learning rate and momentum of stochastic gradient descent optimizer to maximize the
classification accuracy on the MNIST dataset.

This script uses a subset of 4000 samples from the MNIST dataset for quick experimentation.

You can run this example as follows:
    $ python comet_integration.py

Before running this example, make sure you have registered a API key at https://www.comet.com and run
    $comet login

After the script finishes, you can view the optimization results in the comet dashboard.
"""



import optuna
from optuna.integration.comet import CometCallback

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical


TEST_SIZE = 0.25
BATCHSIZE = 16
EPOCHS = 100


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    sample_size = 4000
    test_size = 1000
    train_indices = np.random.choice(x_train.shape[0], sample_size, replace=False)
    test_indices = np.random.choice(x_test.shape[0], test_size, replace=False)
    
    x_train = x_train[train_indices].astype("float32") / 255.0
    x_test = x_test[test_indices].astype("float32") / 255.0
    y_train = to_categorical(y_train[train_indices], 10)
    y_test = to_categorical(y_test[test_indices], 10)
    
    return x_train, y_train, x_test, y_test

def create_model(trial):
    model = Sequential()
    model.add(Input(shape=(28, 28)))
    model.add(Flatten())
    
    n_units = trial.suggest_int("n_units", 32, 256)
    n_layers = trial.suggest_int("n_layers", 1,10)
    for _ in range(n_layers):
        model.add(Dense(n_units, activation="relu", kernel_initializer="normal"))
        
    model.add(Dense(10, activation="softmax"))

    optimizer = SGD(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        momentum=trial.suggest_float("momentum", 0.0, 1.0),
    )
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model



if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize"])
    comet_callback = CometCallback(
        study,
        metric_names=["accuracy"],
        project_name="example-project",
        workspace="example-workspace",
    )
    
    @comet_callback.track_in_comet()
    def objective(trial):

        X_train, y_train, X_valid, y_valid = load_mnist()

        model = create_model(trial)
        model.fit(X_train, y_train, shuffle=True, batch_size=BATCHSIZE, epochs=EPOCHS, verbose=False)

        return model.evaluate(X_valid, y_valid, verbose=0)
    
    
    study.optimize(objective, n_trials=20, callbacks=[comet_callback])
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))