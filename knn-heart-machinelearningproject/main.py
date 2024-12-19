from knn import knn_predict, weighted_knn_predict, evaluate_model
import pandas as pd
import config
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# Load and normalize data
data = pd.read_csv("data.csv").values.tolist()
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform([row[:-1] for row in data])  # Normalize only features
labels = [row[-1] for row in data]  # Extract labels
data = [list(row) + [label] for row, label in zip(scaled_features, labels)]  # Combine scaled features with labels

# Split data into training and testing sets
train_size = int(len(data) * config.train_test_rate)
X_train, y_train = [row[:-1] for row in data[:train_size]], [int(row[-1]) for row in data[:train_size]]
X_test, y_true = [row[:-1] for row in data[train_size:]], [int(row[-1]) for row in data[train_size:]]

# Debugging: Check label distribution
print("Training labels distribution:", Counter(y_train))
print("Test labels distribution:", Counter(y_true))

# Choose which KNN function to use
use_weighted_knn = True  # Change to False if you want to use standard KNN

if use_weighted_knn:
    print("Using Weighted KNN")
    y_pred = weighted_knn_predict(X_train, y_train, X_test, config.k)
else:
    print("Using Standard KNN")
    y_pred = knn_predict(X_train, y_train, X_test, config.k)

# Evaluate the predictions
accuracy, precision, recall, conf_matrix = evaluate_model(y_true, y_pred)

# Debugging: Check predictions
print("y_true sample:", y_true[:10])
print("y_pred sample:", y_pred[:10])

# Print results
print(f"K: {config.k}")
print(f"Train Test Split: {config.train_test_rate*100:.1f} - {100 - config.train_test_rate*100:.1f}")
print(f"Train Data Count: {train_size}/{len(data)}")
print(f"Test Data Count: {len(X_test)}/{len(data)}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("Confusion Matrix:")
print(f" [[{conf_matrix[0][0]} {conf_matrix[0][1]}]  -> [TP  FN]")
print(f" [{conf_matrix[1][0]} {conf_matrix[1][1]}]]  -> [FP  TN]")
