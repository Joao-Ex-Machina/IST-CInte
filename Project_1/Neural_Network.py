import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

CLP = {
    'Decrease': (-1, -0.20),
    'Maintain': (-0.19, 0.19),
    'Increase': (0.20, 1)
}

# Function to classify a single value
def classify_value(value):
    for category, (low, high) in CLP.items():
        if low <= value <= high:
            return category
    return "Out of Range"

# Function to classify an entire array
def classify_array(array):
    return [classify_value(val) for val in array]

# Load the data
data = pd.read_csv('Project_1//CINTE24-25_Proj1_SampleData.csv')

X = data.drop(columns=['V_MemoryUsage', 'V_ProcessorLoad','V_InpNetThroughput', 'V_OutNetThroughput', 'V_OutBandwidth','V_Latency','CLPVariation'])
Y = data.iloc[:, 12].values

Y_class = classify_array(Y)
print(Y_class)
# for i in range(len(Y_class)):
#    print(f"Value: {Y[i]}, Class: {Y_class[i]}")

# Data Split Train 70% Test 15% Validation 15%
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

# Data Split Train 70% Test 30%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_class, test_size=0.3, random_state=0)

clf = MLPClassifier(hidden_layer_sizes=(6), activation='logistic',solver='sgd',max_iter= 200).fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

accuracy = clf.score(X_test, Y_test)
confusion_matrix = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'])

for i in range(len(Y_test)):
    print(f"Actual: {Y_test[i]}, Predicted: {Y_pred[i]}")
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_matrix}")
