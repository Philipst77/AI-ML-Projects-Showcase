import numpy as np
import math 
import csv
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

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

def crossValidation():
    return


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
    k = 5
    
    trvectors, labels, tstvectors = load_and_vectorize(train_file, test_file)
    trvectors = trvectors[:1000]
    labels = labels[:1000]
    tstvectors = tstvectors[:100]

    predictions = knn(trvectors, tstvectors, labels, k)
    print("Predicted", len(predictions), "labels.")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for label in predictions:
            writer.writerow([label])


if __name__ == '__main__':
        main()