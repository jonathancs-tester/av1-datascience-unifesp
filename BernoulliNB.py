import pandas as pd
import sys
import matplotlib.pyplot as plt  
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

dataset = sys.argv[1]
label = sys.argv[2] 
test_size = sys.argv[3] 
model_selection = sys.argv[4] 

df = pd.read_csv(dataset)
labels = df[label]
del df[label] # now df contains the observations

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=float(test_size))

# knn = KNeighborsClassifier()
# knn.fit(X_train,y_train)
# y_pred = knn.predict(X_test)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)

if model_selection.lower() in ['holdout']:
    y_pred = bnb.predict(X_test)
elif model_selection.lower() in ['k folds','k_folds']:
    cv = int(input("Insert the number of windows in k-folds (cross validation): "))
    y_pred = cross_val_predict(bnb, X_test, y_test)
else:
    print('Incorrect model selection. Please, you should choice the holdout or k_folds')

with open('output.log', 'w') as f:
    print('Classification accuracy: {0:.5f}%'.format(metrics.accuracy_score(y_test, y_pred) * 100), file=f)
    print(metrics.classification_report(y_test,y_pred, digits=7), file=f)

    cm = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm, file=f)
    metrics.plot_confusion_matrix(bnb, X_test, y_test)
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    plt.title("Confusion Matrix Visualization")
    plt.savefig('confusion_matrix.png')