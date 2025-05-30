import numpy as np
import math 
import csv
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import random 

def load_and_vectorize(train_file, test_file):
    trvectors= []
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

    vectorizer = TfidfVectorizer(max_features=1000)  
    X_train = vectorizer.fit_transform(trvectors).toarray()
    X_test = vectorizer.transform(tstvectors).toarray()
    
    print("Loaded", len(trvectors), "training reviews and", len(tstvectors), "test reviews.")


    return X_train, labels, X_test

def k10foldcrossValidation(trvectors ,labels, k_values):
    combined = list(zip(trvectors, labels))  
    random.shuffle(combined)  
    
    fold_size = len(combined) // 10 
    
    folds = [] 

    for i in range(10):
        start_index = i * fold_size
        end_index = (i + 1) * fold_size
        fold = combined[start_index:end_index]
        folds.append(fold)  
          
    kvalues_accuracies = {} 
    
    for k in k_values:
        total_correct = 0
        total_predictions = 0
        
        for i in range(10):   
            validation = folds[i]
            training = []
            for j in range(10):
                if j != i:
                 for item in folds[j]:
                     training.append(item)
                                 
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
    print(f"Best k based on 10 fold CV: {best_k}")
    return best_k


def euclidean_Distance (x,y):
    distance =0
    for i in range(len(x)):
        distance += (x[i]-y[i]) ** 2
    return math.sqrt(distance)

def knn(trvectors, tstvectors, labels, k ):
    predictions =[] 
    for i in range(len(tstvectors)):
        distance =[]

        for j in range(len(trvectors)):
         dis = euclidean_Distance( tstvectors[i],trvectors[j])
         distance.append((dis,labels[j]))
         
        distance.sort(key= lambda x : x[0])
        kneartest = distance[:k]
        kneartest_labels = [label for (_, label) in kneartest]
        prediction = Counter(kneartest_labels).most_common(1)[0][0]
        predictions.append(prediction)
    return predictions

def main ():
    train_file = 'new_train.csv'
    test_file = 'new_test.csv'
    output_file = 'my_predictions.csv'
    
    
    trvectors, labels, tstvectors = load_and_vectorize(train_file, test_file)
    
    k_values = [1, 3, 5, 7, 9]  
    best_k = k10foldcrossValidation(trvectors, labels, k_values)
    k = best_k

    

    predictions = knn(trvectors, tstvectors, labels, k)
    print("Predicted", len(predictions), "labels.")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for label in predictions:
            writer.writerow([label])


if __name__ == '__main__':
        main()