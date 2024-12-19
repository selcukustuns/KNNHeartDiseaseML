from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from math import sqrt

def euclidean_distance(point1, point2):
    return sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def knn_predict(X_train, y_train, X_test, k):
    return [
        max(set(nearest_labels := [label for _, label in sorted(
            [(euclidean_distance(test_point, x), label) for x, label in zip(X_train, y_train)]
        )[:k]]), key=nearest_labels.count)
        for test_point in X_test
    ]

def weighted_knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [(euclidean_distance(test_point, x), label) for x, label in zip(X_train, y_train)]
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        
        weights = {label: 0 for label in set(y_train)}
        for dist, label in k_nearest:
            weights[label] += 1 / (dist + 1e-5)
        
        predictions.append(max(weights, key=weights.get))
    return predictions

def evaluate_model(y_true, y_pred):
    if not y_true or len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return 0.0, 0.0, 0.0, [[0, 0], [0, 0]]
    
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='binary', zero_division=0),
        recall_score(y_true, y_pred, average='binary', zero_division=0),
        confusion_matrix(y_true, y_pred)
    )
