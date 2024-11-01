import pandas as pd
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam  # Use Adam optimizer as found in hyperparameter search

def main():
    # ----------- import data and scaling ----------- #
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_train = df_train[['zloc']].values

    X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_test = df_test[['zloc']].values

    # standardized data with separate scalers
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)

    # ----------- create model with best hyperparameters ----------- #
    model = Sequential()
    model.add(Dense(64, input_dim=4, kernel_initializer='normal', activation='relu'))  # First hidden layer with 64 units
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))               # Second hidden layer with 32 units
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))               # Third hidden layer with 32 units (from 'Dense_2': 'three')
    model.add(Dense(1, kernel_initializer='normal'))                                   # Output layer

    # Compile model using Adam optimizer
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mae'])

    # ----------- define callbacks ----------- #
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                       verbose=1, min_delta=1e-5, mode='min')
    modelname = "model@{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(modelname))

    # ----------- start training with optimal batch size ----------- #
    history = model.fit(X_train, y_train,
                        validation_split=0.1, epochs=5000, batch_size=1024,  # Optimal batch size
                        callbacks=[tensorboard, earlyStopping, reduce_lr_loss], verbose=1)

    # ----------- save model and weights ----------- #
    model_json = model.to_json()
    with open("generated_files/{}.json".format(modelname), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("generated_files/{}.weights.h5".format(modelname))
    print("Saved model to disk")

if __name__ == '__main__':
    main()
