import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load Excel file
file_path = r"G:\Godey\Genetic_Algorithm\Deep_Learning\Tyredatastock.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)

train_data = []
test_data = []

# Required column names
required_columns = ['OD', 'TAW', 'INF', 'ORIENT', 'CONT',
                    'X_Undeformed', 'Y_Undeformed', 'X_Deformed', 'Y_Deformed']

for sheet_name, df in sheets.items():
    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: x.strip())

    if not set(required_columns).issubset(df.columns):
        print(f"Skipping sheet '{sheet_name}' due to missing columns.")
        continue

    for i in range(0, len(df), 34):
        block = df.iloc[i:i+34].copy()
        if block.shape[0] < 34:
            continue

        try:
            od_value = float(block.iloc[0]['OD'])
            input_features = block.iloc[0][['OD', 'TAW', 'INF', 'ORIENT', 'CONT']].astype(float).values
            output_coords = block[['X_Undeformed', 'Y_Undeformed', 'X_Deformed', 'Y_Deformed']].astype(float).values.flatten()
            full_data = np.concatenate([input_features, output_coords])

            if 602 <= od_value <= 605:
                train_data.append(full_data)
            if 605 <= od_value <= 607:
                test_data.append(full_data)
        except Exception as e:
            print(f"Skipping block at index {i} in sheet '{sheet_name}' due to error: {e}")
            continue

# Convert to NumPy arrays
train_data = np.array(train_data)
test_data = np.array(test_data)

# Inputs and outputs
X_train_raw, y_train_raw = train_data[:, :5], train_data[:, 5:]
X_test_raw, y_test_raw = test_data[:, :5], test_data[:, 5:]

# Normalize
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train_raw)
y_train = scaler_Y.fit_transform(y_train_raw)

X_test = scaler_X.transform(X_test_raw)
y_test = scaler_Y.transform(y_test_raw)

# Build and train model
model = Sequential([
    Dense(128, input_dim=5, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='linear')
])

model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# Predict on test data
y_pred_scaled = model.predict(X_test)
y_pred = scaler_Y.inverse_transform(y_pred_scaled)

# Save predictions
with open('predictions.txt', 'w') as f:
    for i, prediction in enumerate(y_pred):
        f.write(f'Config {i+1}:\n')
        coords = prediction.reshape(-1, 4)
        for node in coords:
            f.write(' '.join(f'{val:.6f}' for val in node) + '\n')
        f.write('\n')

# Save training data
with open('training_data.txt', 'w') as f_train:
    for i, (x, y) in enumerate(zip(X_train_raw, y_train_raw)):
        f_train.write(f'Train Sample {i+1}:\n')
        f_train.write('Inputs: ' + ' '.join(f'{val:.6f}' for val in x) + '\n')
        coords = y.reshape(-1, 4)
        f_train.write('Outputs:\n')
        for node in coords:
            f_train.write(' '.join(f'{val:.6f}' for val in node) + '\n')
        f_train.write('\n')

# Save testing data
with open('testing_data.txt', 'w') as f_test:
    for i, (x, y) in enumerate(zip(X_test_raw, y_test_raw)):
        f_test.write(f'Test Sample {i+1}:\n')
        f_test.write('Inputs: ' + ' '.join(f'{val:.6f}' for val in x) + '\n')
        coords = y.reshape(-1, 4)
        f_test.write('Outputs:\n')
        for node in coords:
            f_test.write(' '.join(f'{val:.6f}' for val in node) + '\n')
        f_test.write('\n')

# --- CUSTOM PREDICTION ---

# Input values: OD, TAW, INF, ORIENT, CONT 
custom_input = np.array([[603, 105, 0.23, 50, 6.55]])

# Normalize and predict
custom_input_scaled = scaler_X.transform(custom_input)
custom_pred_scaled = model.predict(custom_input_scaled)
custom_pred = scaler_Y.inverse_transform(custom_pred_scaled)
predicted_coords = custom_pred.reshape(-1, 4)

# Save to text file
with open('custom_prediction.txt', 'w') as f:
    f.write(f'Input Parameters: OD={custom_input[0][0]}, TAW={custom_input[0][1]}, INF={custom_input[0][2]}, '
            f'ORIENT={custom_input[0][3]}, CONT={custom_input[0][4]}\n\n')
    f.write('Predicted Coordinates:\n')
    for i, coord in enumerate(predicted_coords):
        x_und, y_und, x_def, y_def = coord
        f.write(f'Node {i+1}: X_Undeformed={x_und:.6f}, Y_Undeformed={y_und:.6f}, '
                f'X_Deformed={x_def:.6f}, Y_Deformed={y_def:.6f}\n')

print("Training data, testing data, predictions, and custom prediction have been saved.")