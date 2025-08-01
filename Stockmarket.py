import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tqdm import tqdm  # Import tqdm for the progress bar
from tensorflow.random import set_seed

# Set random seed for reproducibility
set_seed(455)
np.random.seed(455)

# Full path to Excel file (replace with your file path)
file_path = r"G:\Godey\Genetic_Algorithm\Deep_Learning\NIFTY50DUMMY.xlsx"  # Update the path

# Load and clean the dataset
dataset = pd.read_excel(file_path)
dataset.columns = dataset.columns.str.strip()  # Remove any leading/trailing spaces in columns
dataset["Date"] = pd.to_datetime(dataset["Date"])
dataset.set_index("Date", inplace=True)
dataset.drop(["Turnover"], axis=1, inplace=True)  # Drop "Turnover" column if not needed
print(dataset.head())
print(dataset.describe())
print(dataset.isna().sum())

# Train-test split plotting
tstart = 2016
tend = 2020

def train_test_plot(dataset, tstart, tend):
    dataset.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[f"{tend+1}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {tend+1})", f"Test ({tend+1} and beyond)"])
    plt.title("NIFTY 50 High Price")
    plt.show()

train_test_plot(dataset, tstart, tend)

def train_test_split(dataset, tstart, tend):
    train = dataset.loc[f"{tstart}":f"{tend}", "High"].values
    test = dataset.loc[f"{tend+1}":, "High"].values
    return train, test

training_set, test_set = train_test_split(dataset, tstart, tend)

# Scale training data
sc = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1)
training_set_scaled = sc.fit_transform(training_set)

# Sequence splitting function
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:i + n_steps])
        y.append(sequence[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 60
features = 1

X_train, y_train = split_sequence(training_set_scaled, n_steps)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer="RMSprop", loss="mse")
model_lstm.summary()

# Training the model with progress bar using tqdm
epochs = 50  # You can adjust this value for training
for epoch in tqdm(range(epochs), desc="Training LSTM Model", unit="epoch"):
    history = model_lstm.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
    if epoch % 5 == 0:  # Print loss every 5 epochs
        print(f"Epoch {epoch}/{epochs}, Loss: {history.history['loss'][0]}")

# Prepare test data
dataset_total = dataset["High"]
inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test, y_test = split_sequence(inputs, n_steps)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], features))

# Predict stock prices
predicted_stock_price = model_lstm.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting function
def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("NIFTY 50 High Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# RMSE Calculation
def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}".format(rmse))

# Evaluate the model performance
plot_predictions(test_set[:len(predicted_stock_price)], predicted_stock_price)
return_rmse(test_set[:len(predicted_stock_price)], predicted_stock_price)