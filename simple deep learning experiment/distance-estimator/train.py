# import pandas as pd
# import numpy as np
# import time

# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.preprocessing import StandardScaler
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
# from keras.optimizers import Adam  # Use Adam optimizer as found in hyperparameter search

# def main():
#     # ----------- import data and scaling ----------- #
#     df_train = pd.read_csv('data/train.csv')
#     df_test = pd.read_csv('data/test.csv')

#     X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
#     y_train = df_train[['zloc']].values

#     X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
#     y_test = df_test[['zloc']].values

#     # standardized data with separate scalers
#     X_scaler = StandardScaler()
#     y_scaler = StandardScaler()

#     X_train = X_scaler.fit_transform(X_train)
#     y_train = y_scaler.fit_transform(y_train)

#     # ----------- create model with best hyperparameters ----------- #
#     model = Sequential()
#     model.add(Dense(64, input_dim=4, kernel_initializer='normal', activation='relu'))  # First hidden layer with 64 units
#     model.add(Dense(32, kernel_initializer='normal', activation='relu'))               # Second hidden layer with 32 units
#     model.add(Dense(32, kernel_initializer='normal', activation='relu'))               # Third hidden layer with 32 units (from 'Dense_2': 'three')
#     model.add(Dense(1, kernel_initializer='normal'))                                   # Output layer

#     # Compile model using Adam optimizer
#     model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mae'])

#     # ----------- define callbacks ----------- #
#     earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
#     reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
#                                        verbose=1, min_delta=1e-5, mode='min')
#     modelname = "model@{}".format(int(time.time()))
#     tensorboard = TensorBoard(log_dir="logs/{}".format(modelname))

#     # ----------- start training with optimal batch size ----------- #
#     history = model.fit(X_train, y_train,
#                         validation_split=0.1, epochs=5000, batch_size=1024,  # Optimal batch size
#                         callbacks=[tensorboard, earlyStopping, reduce_lr_loss], verbose=1)

#     # ----------- save model and weights ----------- #
#     model_json = model.to_json()
#     with open("generated_files/{}.json".format(modelname), "w") as json_file:
#         json_file.write(model_json)

#     model.save_weights("generated_files/{}.weights.h5".format(modelname))
#     print("Saved model to disk")

# if __name__ == '__main__':
#     main()

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import csv

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras.optimizers import RMSprop  # Import RMSprop optimizer

def within_5_percent(y_true, y_pred):
    margin = 0.05
    within_margin = np.abs((y_pred - y_true) / y_true) <= margin
    return np.mean(within_margin)

def main():
    # ----------- Import data and perform scaling ----------- #
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_train = df_train[['zloc']].values

    X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_test = df_test[['zloc']].values

    # Standardize data with separate scalers
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)

    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)

    # ----------- Create model with best hyperparameters ----------- #
    # Updated hyperparameters based on tuning results
    layer_1_units = 32
    layer_2_units = 64
    layer_3_units = 16
    batch_size = 1024
    optimizer = Adam()

    # layer_1_units = 128
    # layer_2_units = 64
    # layer_3_units = 32
    # batch_size = 512
    # optimizer = RMSprop()

    model = Sequential()
    model.add(Dense(layer_1_units, input_dim=4, activation='relu'))  # First hidden layer
    model.add(Dense(layer_2_units, activation='relu'))               # Second hidden layer
    model.add(Dense(layer_3_units, activation='relu'))               # Third hidden layer

    # # Check if the third layer is required (according to the hyperparameters result)
    # if layer_3_units is not None:
    #     model.add(Dense(layer_3_units, activation='relu'))

    # model.add(Dense(1, kernel_initializer='normal')) # Output layer
    model.add(Dense(1)) # Output layer

    # Compile model using Adam optimizer
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    # ----------- Define callbacks ----------- #
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                       verbose=1, min_delta=1e-5, mode='min')
    modelname = "model@{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="logs/{}".format(modelname))

    # ----------- Start training with optimal batch size ----------- #
    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        epochs=5000,
                        batch_size=batch_size,
                        callbacks=[tensorboard, earlyStopping, reduce_lr_loss],
                        verbose=1)

    # ----------- Save model and weights ----------- #
    model_json = model.to_json()
    with open("generated_files/{}.json".format(modelname), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("generated_files/{}.weights.h5".format(modelname))
    print("Saved model to disk")

    # ----------- Save loss and metrics to CSV ----------- #
    # Save training and validation loss for each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(loss) + 1)

    # Save to CSV
    with open('generated_files/{}_training_log_5.csv'.format(modelname), 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss', 'val_loss', 'mae', 'val_mae']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(epochs)):
            writer.writerow({
                'epoch': epochs[i],
                'loss': loss[i],
                'val_loss': val_loss[i],
                'mae': mae[i],
                'val_mae': val_mae[i]
            })

    # ----------- Visualize training ----------- #
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('generated_files/{}_loss_plot_5.png'.format(modelname))
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, mae, label='Training MAE')
    plt.plot(epochs, val_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.savefig('generated_files/{}_mae_plot_5.png'.format(modelname))
    plt.show()

    # ----------- Evaluate on test data ----------- #
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print('Test Loss:', test_loss)
    print('Test MAE:', test_mae)

    # ----------- Calculate 5% margin accuracy ----------- #
    y_pred = model.predict(X_test)
    y_test_inv = y_scaler.inverse_transform(y_test)
    y_pred_inv = y_scaler.inverse_transform(y_pred)
    five_percent_accuracy = within_5_percent(y_test_inv, y_pred_inv) * 100
    print('Percentage of predictions within 5% margin:', five_percent_accuracy, '%')

    # ----------- Save configuration and metrics to CSV ----------- #
    with open('generated_files/{}_final_results_5.csv'.format(modelname), 'w', newline='') as csvfile:
        fieldnames = ['Hidden Layer 1', 'Hidden Layer 2', 'Hidden Layer 3', 
                      'Batch Size', 'Optimizer', 'Validation MSE', 'Validation MAE', 
                      '5% Margin Accuracy', 'Loss History', 'Validation Loss History']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'Hidden Layer 1': layer_1_units,
            'Hidden Layer 2': layer_2_units,
            'Hidden Layer 3': layer_3_units,
            'Batch Size': batch_size,
            'Optimizer': optimizer,
            'Validation MSE': test_loss,
            'Validation MAE': test_mae,
            '5% Margin Accuracy': five_percent_accuracy,
            'Loss History': ','.join(map(str, loss)),
            'Validation Loss History': ','.join(map(str, val_loss))
        })

if __name__ == '__main__':
    main()

