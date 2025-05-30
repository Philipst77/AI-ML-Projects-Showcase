import numpy as np
import math 
import csv
from collections import Counter
import random 
random.seed(42)
np.random.seed(42)


def load_and_vectorize(train_file, test_file):
    trvectors = []
    labels = []

    with open(train_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(int(row[0]))
            trvectors.append(row[1])

    tstvectors = []
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            tstvectors.append(row[0])

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(trvectors).toarray()
    X_test = vectorizer.transform(tstvectors).toarray()

    print("Loaded", len(trvectors), "training reviews and", len(tstvectors), "test reviews.")

    return X_train, labels, X_test


def cosine_distance_np(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1 - (dot / (norm1 * norm2))


def knn(trvectors, tstvectors, labels, k):
    predictions = []

    trvectors = np.array(trvectors)
    tstvectors = np.array(tstvectors)
    labels = np.array(labels)

    train_norms = np.linalg.norm(trvectors, axis=1)

    for test_vec in tstvectors:
        test_norm = np.linalg.norm(test_vec)
        sims = []

        for i in range(len(trvectors)):
            dot = np.dot(test_vec, trvectors[i])
            if test_norm == 0 or train_norms[i] == 0:
                sim = -1
            else:
                sim = dot / (test_norm * train_norms[i])
            sims.append(sim)

        top_k_idx = np.argsort(sims)[-k:]
        top_k_labels = [labels[i] for i in top_k_idx]
        prediction = Counter(top_k_labels).most_common(1)[0][0]
        predictions.append(prediction)

    return predictions


def k10foldcrossValidation(trvectors, labels, k_values):
    combined = list(zip(trvectors, labels))
    fold_size = len(combined) // 10
    folds = []

    for i in range(10):
        start_index = i * fold_size
        end_index = (i + 1) * fold_size
        folds.append(combined[start_index:end_index])

    kvalues_accuracies = {}

    for k in k_values:
        total_correct = 0
        total_predictions = 0

        for i in range(10):
            validation = folds[i]
            training = [item for j in range(10) if j != i for item in folds[j]]

            train_vectors = [x[0] for x in training]
            train_labels = [x[1] for x in training]
            val_vectors = [x[0] for x in validation]
            val_labels = [x[1] for x in validation]

            predictions = knn(train_vectors, val_vectors, train_labels, k)

            correct = sum(1 for pred, actual in zip(predictions, val_labels) if pred == actual)
            total_correct += correct
            total_predictions += len(val_labels)

        accuracy = total_correct / total_predictions
        kvalues_accuracies[k] = accuracy
        print(f"k={k}, accuracy={accuracy:.4f}")

    best_k = max(kvalues_accuracies, key=kvalues_accuracies.get)
    print(f"Best k based on 10-fold CV: {best_k}")
    return best_k


def main():
    train_file = 'new_train.csv'
    test_file = 'new_test.csv'
    output_file = 'my_predictions.csv'

    trvectors, labels, tstvectors = load_and_vectorize(train_file, test_file)



    combined = list(zip(trvectors, labels))
    random.shuffle(combined)
    trvectors, labels = zip(*combined)
    trvectors = np.array(trvectors)
    labels = np.array(labels)

    k_values = [1, 3, 5, 7, 9]
    best_k = k10foldcrossValidation(trvectors, labels, k_values)

    predictions = knn(trvectors, tstvectors, labels, best_k)
    print("Predicted", len(predictions), "labels.")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for label in predictions:
            writer.writerow([label])


if __name__ == '__main__':
    main()
