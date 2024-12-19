# Weighted KNN Classifier with Normalization

This project implements a weighted K-Nearest Neighbors (KNN) algorithm with additional features like data normalization and model evaluation metrics. The project is designed for simplicity and modularity, allowing easy configuration and use.

## Features
- **Data Normalization**: Uses `MinMaxScaler` to normalize feature columns to a 0-1 range.
- **Weighted KNN**: Calculates weights for neighbors based on their distance to improve prediction accuracy.
- **Configurable Parameters**: Configurable number of neighbors (`k`) and train-test split ratio via the `config.py` file.
- **Model Evaluation**: Provides metrics such as accuracy, precision, recall, and a confusion matrix.

## File Structure
- **`data.csv`**: Dataset file containing features and labels.
- **`knn.py`**: Core module implementing:
  - Euclidean distance calculation.
  - Basic KNN prediction.
  - Model evaluation functions.
- **`config.py`**: Configuration file for defining hyperparameters like `k` and train-test split ratio.
- **`main.py`**: Main script that orchestrates the data loading, normalization, training, prediction, and evaluation processes.

## Requirements
- Python 3.7 or above
- Required libraries:
  - `pandas`
  - `scikit-learn`

Install the required libraries using pip:
```bash
pip install pandas scikit-learn
