import time
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice

def data():
    # ----------- import data and perform scaling ----------- #
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_train = df_train[['zloc']].values

    X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_test = df_test[['zloc']].values

    # Fit scaler on training data only
    scalar_X = StandardScaler()
    scalar_y = StandardScaler()

    x_train = scalar_X.fit_transform(X_train)
    y_train = scalar_y.fit_transform(y_train)

    # Transform test data with scalers fitted on training data
    x_test = scalar_X.transform(X_test)
    y_test = scalar_y.transform(y_test)

    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    # ----------- define model ----------- #
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Sequential()
        
        # Input layer
        model.add(Dense(6, input_shape=(4,), activation='relu'))
        
        # Experimenting with different hidden layer sizes and depths
        model.add(Dense({{choice([32, 64, 128])}}, activation='relu'))
        model.add(Dense({{choice([16, 32, 64])}}, activation='relu'))

        # Optionally adding a third hidden layer
        if {{choice(['two', 'three'])}} == 'three':
            model.add(Dense({{choice([16, 32])}}, activation='relu'))
        
        # Output layer
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', metrics=['mae'],
                      optimizer={{choice(['rmsprop', 'adam'])}})

    # ----------- define callbacks ----------- #
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                       verbose=1, min_delta=1e-4, mode='min')
    tensorboard = TensorBoard(log_dir="logs/model@{}".format(int(time.time())))

    # ----------- start training ----------- #
    model.fit(x_train, y_train,
              batch_size={{choice([512, 1024, 2048])}},
              epochs=5000,
              callbacks=[tensorboard, earlyStopping, reduce_lr_loss],
              verbose=1,
              validation_split=0.1)

    # ----------- evaluate model ----------- #
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test MAE:', score[1])

    return {'loss': score[0], 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=trials,
                                          eval_space=True)

    X_train, Y_train, X_test, Y_test = data()
    print("Evaluation of best performing model:", best_model.evaluate(X_test, Y_test))
    print("Best hyperparameters found:", best_run)
