import pandas as pd
import argparse
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

argparser = argparse.ArgumentParser(description='Get predictions of test set')
argparser.add_argument('-m', '--modelname',
                       help='model name (.json)')
argparser.add_argument('-w', '--weights',
                       help='weights filename (.h5)')

args = argparser.parse_args()

# Parse arguments
MODEL = args.modelname
WEIGHTS = args.weights

def main():
    # Load the test data
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

    # Transform the test data with the preprocessor used during training
    X_test = preprocessor.fit_transform(df_test)
    y_test = df_test[['zloc']].values

    # Standardize the target variable (zloc) using the scaler used during training
    scaler_y = StandardScaler()
    y_test = scaler_y.fit_transform(y_test)

    # Load JSON and create model
    json_file = open(f'generated_files/{MODEL}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into the model
    loaded_model.load_weights(f"generated_files/{WEIGHTS}.h5")
    print("Loaded model from disk")

    # Compile the model (the optimizer should match the training setup)
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')

    # Make predictions
    y_pred = loaded_model.predict(X_test)

    # Inverse transform the scaled predictions and actual values to original scale
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    # Save predictions to CSV
    df_result = df_test.copy()
    df_result['zloc_pred'] = y_pred_inv.flatten()  # Flatten to match original shape
    df_result.to_csv(f'data/predictions_{MODEL}.csv', index=False)

    print(f"Predictions saved to data/predictions_{MODEL}.csv")

if __name__ == '__main__':
    main()
