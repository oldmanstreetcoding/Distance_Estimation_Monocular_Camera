# import time
# import pandas as pd
# import numpy as np
# import csv
# import matplotlib.pyplot as plt

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.regularizers import l2
# from keras.optimizers import Adam, RMSprop
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
# import tensorflow as tf

# # Set up data loading and preprocessing
# def data():
#     # Load data
#     df_train = pd.read_csv('data/train.csv')
#     df_test = pd.read_csv('data/test.csv')

#     # Define numeric and categorical features
#     numeric_features = ['xmin', 'ymin', 'xmax', 'ymax']
#     categorical_features = ['class']

#     # Set up a column transformer for scaling numeric features and one-hot encoding the class feature
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(), categorical_features)
#         ])

#     # Fit the transformer on training data and transform both training and testing data
#     X_train = preprocessor.fit_transform(df_train)
#     y_train = df_train[['zloc']].values
#     X_test = preprocessor.transform(df_test)
#     y_test = df_test[['zloc']].values

#     # Standardize the target variable (zloc)
#     scaler_y = StandardScaler()
#     y_train = scaler_y.fit_transform(y_train)
#     y_test = scaler_y.transform(y_test)

#     return X_train, y_train, X_test, y_test, scaler_y

# # Function to experiment with different model configurations
# def build_model(layers=3, activation='relu', dropout_rate=0.2, l2_lambda=0.01, use_batch_norm=False, optimizer='adam', learning_rate=0.001):
#     # Set up the optimizer with the specified learning rate
#     if optimizer == 'adam':
#         opt = Adam(learning_rate=learning_rate)
#     elif optimizer == 'rmsprop':
#         opt = RMSprop(learning_rate=learning_rate)
#     else:
#         raise ValueError("Unsupported optimizer")

#     # Define the model architecture
#     model = Sequential()
#     input_dim = X_train.shape[1]  # Input dimension based on preprocessed data

#     # First hidden layer
#     model.add(Dense(128, input_dim=input_dim, activation=activation, kernel_regularizer=l2(l2_lambda)))
#     if use_batch_norm:
#         model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))

#     # Additional hidden layers
#     for _ in range(layers - 2):
#         model.add(Dense(64, activation=activation, kernel_regularizer=l2(l2_lambda)))
#         if use_batch_norm:
#             model.add(BatchNormalization())
#         model.add(Dropout(dropout_rate))

#     # Last hidden layer
#     model.add(Dense(16, activation=activation, kernel_regularizer=l2(l2_lambda)))
#     if use_batch_norm:
#         model.add(BatchNormalization())

#     # Output layer
#     model.add(Dense(1))

#     # Compile model
#     model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])

#     return model

# # Training and evaluation function
# def train_and_evaluate(X_train, y_train, X_test, y_test, scaler_y, config):
#     # Get configuration parameters
#     layers = config.get('layers', 3)
#     activation = config.get('activation', 'relu')
#     dropout_rate = config.get('dropout_rate', 0.2)
#     l2_lambda = config.get('l2_lambda', 0.01)
#     use_batch_norm = config.get('use_batch_norm', False)
#     optimizer = config.get('optimizer', 'adam')
#     learning_rate = config.get('learning_rate', 0.001)
#     batch_size = config.get('batch_size', 1024)
#     epochs = config.get('epochs', 500)

#     # Build the model with the specified configuration
#     model = build_model(layers=layers, activation=activation, dropout_rate=dropout_rate, 
#                         l2_lambda=l2_lambda, use_batch_norm=use_batch_norm, 
#                         optimizer=optimizer, learning_rate=learning_rate)

#     # Callbacks
#     early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=1e-4, mode='min')
#     tensorboard = TensorBoard(log_dir="logs/model@{}".format(int(time.time())))

#     # Train the model
#     history = model.fit(X_train, y_train, 
#                         validation_split=0.1, 
#                         epochs=epochs, 
#                         batch_size=batch_size,
#                         callbacks=[tensorboard, early_stopping, reduce_lr], 
#                         verbose=1)

#     # Evaluate the model on the test set
#     test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
#     print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

#     # Calculate 5% margin accuracy
#     y_pred = model.predict(X_test)
#     y_test_inv = scaler_y.inverse_transform(y_test)
#     y_pred_inv = scaler_y.inverse_transform(y_pred)
#     five_percent_accuracy = np.mean(np.abs(y_test_inv - y_pred_inv) / y_test_inv <= 0.05) * 100
#     print(f'5% Margin Accuracy: {five_percent_accuracy}%')

#     return history, test_loss, test_mae, five_percent_accuracy

# # Configuration list to try different strategies
# configs = [
#     {'layers': 3, 'activation': 'relu', 'dropout_rate': 0.2, 'l2_lambda': 0.01, 'use_batch_norm': True, 'optimizer': 'adam', 'learning_rate': 0.001, 'batch_size': 512, 'epochs': 200},
#     {'layers': 3, 'activation': 'leaky_relu', 'dropout_rate': 0.3, 'l2_lambda': 0.01, 'use_batch_norm': False, 'optimizer': 'adam', 'learning_rate': 0.0005, 'batch_size': 1024, 'epochs': 300},
#     {'layers': 4, 'activation': 'elu', 'dropout_rate': 0.25, 'l2_lambda': 0.01, 'use_batch_norm': True, 'optimizer': 'rmsprop', 'learning_rate': 0.001, 'batch_size': 2048, 'epochs': 400}
# ]

# # Load data
# X_train, y_train, X_test, y_test, scaler_y = data()

# # Experiment with each configuration
# for i, config in enumerate(configs):
#     print(f"\n--- Experiment {i+1} with Config: {config} ---")
#     history, test_loss, test_mae, five_percent_accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, scaler_y, config)
    
#     # Plot results
#     plt.figure(figsize=(12, 6))
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title(f"Experiment {i+1} - Loss")
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(history.history['mae'], label='Training MAE')
#     plt.plot(history.history['val_mae'], label='Validation MAE')
#     plt.title(f"Experiment {i+1} - MAE")
#     plt.xlabel('Epochs')
#     plt.ylabel('MAE')
#     plt.legend()
#     plt.show()

#     print(f"Results for Experiment {i+1}:")
#     print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, 5% Margin Accuracy: {five_percent_accuracy}%\n")

# print("Experiments completed.")

import pandas as pd
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import RMSprop

# Helper function to calculate 5% margin accuracy
def within_5_percent(y_true, y_pred):
    margin = 0.05
    within_margin = np.abs((y_pred - y_true) / y_true) <= margin
    return np.mean(within_margin)

# Step 1: Data Loading and Preprocessing
def load_data():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    numeric_features = ['xmin', 'ymin', 'xmax', 'ymax']
    categorical_features = ['class']

    # Preprocessing: Scaling numeric and one-hot encoding categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    X_train = preprocessor.fit_transform(df_train)
    y_train = df_train[['zloc']].values
    X_test = preprocessor.transform(df_test)
    y_test = df_test[['zloc']].values

    # Scaling the target variable (zloc)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return X_train, y_train, X_test, y_test, y_scaler

# Step 2: Building the Combined Model
def build_combined_model(input_dim):
    model = Sequential()
    
    # First hidden layer with dropout and batch normalization
    model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Second hidden layer with dropout and batch normalization
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Third hidden layer with LeakyReLU activation and batch normalization
    model.add(Dense(16, activation='linear', kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=0.01))  # Using LeakyReLU
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1))

    # Compile the model with RMSprop and a reduced learning rate
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model

# Step 3: Training and Evaluation
def main():
    # Load and preprocess data
    X_train, y_train, X_test, y_test, y_scaler = load_data()

    # Model parameters
    input_dim = X_train.shape[1]
    batch_size = 128
    epochs = 200

    # Build model
    model = build_combined_model(input_dim)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=1e-4, mode='min')
    tensorboard = TensorBoard(log_dir="logs/model@{}".format(int(time.time())))

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tensorboard, early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate model on test data
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print("Test Loss:", test_loss)
    print("Test MAE:", test_mae)

    # Calculate 5% margin accuracy
    y_pred = model.predict(X_test)
    y_test_inv = y_scaler.inverse_transform(y_test)
    y_pred_inv = y_scaler.inverse_transform(y_pred)
    five_percent_accuracy = within_5_percent(y_test_inv, y_pred_inv) * 100
    print("5% Margin Accuracy:", five_percent_accuracy, "%")

    # Save training history to CSV
    model_name = "model@{}".format(int(time.time()))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(loss) + 1)

    with open('generated_files/{}_training_log.csv'.format(model_name), 'w', newline='') as csvfile:
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

    # Save model architecture and weights
    model_json = model.to_json()
    with open("generated_files/{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("generated_files/{}.weights.h5".format(model_name))
    print("Model saved to disk.")

    # Save final results to CSV
    with open('generated_files/{}_final_results.csv'.format(model_name), 'w', newline='') as csvfile:
        fieldnames = ['Hidden Layer 1', 'Hidden Layer 2', 'Hidden Layer 3', 
                      'Batch Size', 'Optimizer', 'Validation MSE', 'Validation MAE', 
                      '5% Margin Accuracy', 'Loss History', 'Validation Loss History']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'Hidden Layer 1': 128,
            'Hidden Layer 2': 64,
            'Hidden Layer 3': 16,
            'Batch Size': batch_size,
            'Optimizer': 'RMSprop',
            'Validation MSE': test_loss,
            'Validation MAE': test_mae,
            '5% Margin Accuracy': five_percent_accuracy,
            'Loss History': ','.join(map(str, loss)),
            'Validation Loss History': ','.join(map(str, val_loss))
        })

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('generated_files/{}_loss_plot.png'.format(model_name))
    plt.show()

    # Plot training and validation MAE
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, mae, label='Training MAE')
    plt.plot(epochs, val_mae, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.savefig('generated_files/{}_mae_plot.png'.format(model_name))
    plt.show()

if __name__ == '__main__':
    main()

