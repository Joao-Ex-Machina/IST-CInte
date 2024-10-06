import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import recall_score, classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

CLP = {
    'Decrease': (-1, -0.34),
    'Maintain': (-0.33, 0.33),
    'Increase': (0.34, 1)
}

def classify_value(value):
    value = round(value, 2)  # Round to 2 decimal places
    for category, (low, high) in CLP.items():
        if low <= value <= high:
            return category
    print("Value: ", value)
    return "Out of Range"

# Function to classify an entire array
def classify_array(array):
    return [classify_value(val) for val in array]

# Load the data
data = pd.read_csv('Project_1//src//randomset.csv')
data.columns = ["Memory","Processor","Input","Output","Bandwidth","Latency","CLP"]
X = data.iloc[:, 0:6].values
Y = data.iloc[:, 6].values

Y_class = classify_array(Y)
class_counts = Counter(Y_class)
print(f"Class counts: {class_counts}")

# for i in range(len(Y_class)):
#    print(f"Value: {Y[i]}, Class: {Y_class[i]}")
Validation = False

if(Validation == True):
    # Data Split Train 70% Test 15% Validation 15%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_class, test_size=0.3, random_state=0)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)
if(Validation == False):
    # Data Split Train 70% Test 30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_class, test_size=0.3, random_state=0)

class_counts_train = Counter(Y_train)
print(f"Class counts (Train): {class_counts_train}")

smote = SMOTE(random_state=0)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)

print(f"Balanced class counts: {Counter(Y_train_balanced)}")

if(Validation == False):
    clf = MLPClassifier(hidden_layer_sizes=(10,4), activation='relu',solver='adam',max_iter= 1000).fit(X_train_balanced, Y_train_balanced)
    print (clf.score(X_train,Y_train))
    plt.plot(clf.loss_curve_)
    plt.show()
    
if(Validation == True):
    param_grid = {
        'hidden_layer_sizes': [(10,3),(10,4)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],}

    Best_Accuracy = 0   
    for i in range(param_grid['hidden_layer_sizes'].__len__()):
        for j in range(param_grid['activation'].__len__()):
            for k in range(param_grid['solver'].__len__()):
                clf = MLPClassifier(hidden_layer_sizes=param_grid['hidden_layer_sizes'][i], activation=param_grid['activation'][j],solver=param_grid['solver'][k],max_iter= 1000).fit(X_train_balanced, Y_train_balanced)
                accuracy = clf.score(X_val, Y_val)
                print(f"Accuracy: {accuracy}, Hidden Layer Sizes: {param_grid['hidden_layer_sizes'][i]}, Activation: {param_grid['activation'][j]}, Solver: {param_grid['solver'][k]}")
                if(accuracy > Best_Accuracy):
                    best_param = [param_grid['hidden_layer_sizes'][i],param_grid['activation'][j],param_grid['solver'][k]]

    #Accuracy: 0.9586666666666667, Hidden Layer Sizes: (10, 3), Activation: tanh, Solver: lbfgs
    print(f"Best Parameters: {best_param}")   
    clf = MLPClassifier(hidden_layer_sizes=best_param[0], activation=best_param[1],solver=best_param[2],max_iter= 1000,early_stopping=True).fit(X_train_balanced, Y_train_balanced)               

Y_pred = clf.predict(X_test)

accuracy = clf.score(X_test, Y_test)
confusion_matrix = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'])



print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_matrix}")
report = classification_report(Y_test, Y_pred,zero_division=0)
print(report)
