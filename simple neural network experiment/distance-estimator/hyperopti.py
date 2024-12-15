# adding a new neuron: object type, but didn't give accuracy improvement
import time
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice

# Step 1: Data Loading and Scaling
def data():
    # Load data
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    # Define numeric and categorical features
    numeric_features = ['xmin', 'ymin', 'xmax', 'ymax']
    categorical_features = ['class']

    # Set up a column transformer for scaling numeric features and one-hot encoding the class feature
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Fit the transformer on training data and transform both training and testing data
    X_train = preprocessor.fit_transform(df_train)
    y_train = df_train[['zloc']].values
    X_test = preprocessor.transform(df_test)
    y_test = df_test[['zloc']].values

    # Standardize the target variable (zloc)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    return X_train, y_train, X_test, y_test

# Step 2: Model Definition and Hyperparameter Choices
def create_model(X_train, y_train, X_test, y_test):
    # Helper function to calculate 5% margin accuracy
    def within_5_percent(y_true, y_pred):
        margin = 0.05
        within_margin = np.abs((y_pred - y_true) / y_true) <= margin
        return np.mean(within_margin)

    # Define hyperparameters
    layer_1_units = {{choice([32, 64, 128])}}
    layer_2_units = {{choice([16, 32, 64])}}
    layer_3_units = {{choice([16, 32])}} if {{choice(['two', 'three'])}} == 'three' else None
    batch_size = {{choice([512, 1024, 2048])}}
    optimizer = {{choice(['rmsprop', 'adam'])}}

    # Define model architecture
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Sequential()
        
        # Adjust input layer to match the number of features after one-hot encoding
        input_dim = X_train.shape[1]  # Adjust input size based on the number of columns in X_train
        model.add(Dense(layer_1_units, input_shape=(input_dim,), activation='relu'))
        
        # Hidden layers
        model.add(Dense(layer_2_units, activation='relu'))
        
        if layer_3_units is not None:
            model.add(Dense(layer_3_units, activation='relu'))
        
        # Output layer
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', metrics=['mae'], optimizer=optimizer)

    # Define callbacks
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                       verbose=1, min_delta=1e-4, mode='min')
    tensorboard = TensorBoard(log_dir="logs/model@{}".format(int(time.time())))

    # Train the model
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=100,
                        callbacks=[tensorboard, earlyStopping, reduce_lr_loss],
                        verbose=1,
                        validation_split=0.1)

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test MAE:', score[1])

    # Calculate 5% margin accuracy
    y_pred = model.predict(X_test)
    five_percent_accuracy = within_5_percent(y_test, y_pred)
    print('Percentage of predictions within 5% margin:', five_percent_accuracy * 100, '%')

    # Collect details for the current trial, including loss history
    model_details = {
        'Hidden Layer 1': layer_1_units,
        'Hidden Layer 2': layer_2_units,
        'Hidden Layer 3': layer_3_units if layer_3_units is not None else 'N/A',
        'Batch Size': batch_size,
        'Optimizer': optimizer,
        'Validation MSE': score[0],
        'Validation MAE': score[1],
        '5% Margin Accuracy': five_percent_accuracy * 100,
        'Loss History': history.history['loss'],
        'Validation Loss History': history.history['val_loss']
    }

    # Return model details for logging in the main function
    return {'loss': score[0], 'status': STATUS_OK, 'model': model, 'model_details': model_details}

# Step 3: Run Hyperparameter Optimization
if __name__ == '__main__':
    trials = Trials()
    all_results = []

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=trials,
                                          eval_space=True)

    X_train, Y_train, X_test, Y_test = data()
    print("Evaluation of best performing model:", best_model.evaluate(X_test, Y_test))
    print("Best hyperparameters found:", best_run)

    # Collect model details for each trial and save to CSV
    for trial in trials.trials:
        model_details = trial['result']['model_details']
        all_results.append(model_details)

    with open('tuning/hyperparameter_tuning_results_5b.csv', 'w', newline='') as csvfile:
        fieldnames = ['Hidden Layer 1', 'Hidden Layer 2', 'Hidden Layer 3', 
                      'Batch Size', 'Optimizer', 'Validation MSE', 'Validation MAE', '5% Margin Accuracy', 'Loss History', 'Validation Loss History']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            result['Loss History'] = ','.join(map(str, result['Loss History']))
            result['Validation Loss History'] = ','.join(map(str, result['Validation Loss History']))
            writer.writerow(result)

    # Plotting Results for Comparison
    model_indices = range(1, len(all_results) + 1)
    mse_values = [result['Validation MSE'] for result in all_results]
    mae_values = [result['Validation MAE'] for result in all_results]
    margin_accuracies = [result['5% Margin Accuracy'] for result in all_results]
    
    plt.figure(figsize=(12, 6))
    plt.plot(model_indices, mse_values, label='Validation MSE', marker='o')
    plt.plot(model_indices, mae_values, label='Validation MAE', marker='x')
    plt.plot(model_indices, margin_accuracies, label='5% Margin Accuracy', marker='s')
    plt.xlabel('Model Index')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Hyperparameter Tuning Results with 5% Margin Accuracy')
    plt.savefig('tuning/hyperparameter_tuning_performance_5b.png')
    plt.show()
