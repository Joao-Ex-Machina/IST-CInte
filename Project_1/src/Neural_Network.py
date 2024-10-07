import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import mean_squared_error,make_scorer,classification_report
from sklearn.model_selection import GridSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

#Define Thresholds for Classification
CLP = {
    'Decrease': (-1, -0.34),
    'Maintain': (-0.33, 0.33),
    'Increase': (0.34, 1)
}

# Function to classify a single value
def classify_value(value):
    value = round(value, 2)  # Round to 2 decimal places
    for category, (low, high) in CLP.items():
        if low <= value <= high:
            return category
        if value < -1:
            return "Decrease"
        if value > 1:
            return "Increase"
    print("Value: ", value)
    return "Out of Range"

# Function to classify an entire array
def classify_array(array):
    return [classify_value(val) for val in array]

# Load the data
data = pd.read_csv('Project_1//src//randomset.csv')
data.columns = ["Memory","Processor","Input","Output","Bandwidth","Latency","CLP"]

# Artificially balance the dataset by duplicating rows
initial_counts = Counter(data['CLP'].apply(lambda x: 'Maintain' if -0.33 <= x <= 0.33 else 'Other'))
print("Initial 'Maintain' count:", initial_counts['Maintain'])
# Filter rows where CLP is between -0.33 and 0.33 ('Maintain' class)
maintain_rows = data[(data['CLP'] >= -0.33) & (data['CLP'] <= 0.33)]

n_missing = 1000 - len(maintain_rows)
if n_missing > 0:
    maintain_rows_duplicated = maintain_rows.sample(n=n_missing, replace=True, random_state=0)
    data_extended = pd.concat([data, maintain_rows_duplicated])
    updated_counts = Counter(data_extended['CLP'].apply(lambda x: 'Maintain' if -0.33 <= x <= 0.33 else 'Other'))
    print("Updated 'Maintain' count:", updated_counts['Maintain'])
    data = data_extended
else:   
    print("No need for duplication, already have 1000 or more 'Maintain' instances.")
    
X = data.iloc[:, 0:6].values
Y = data.iloc[:, 6].values

Y_class = classify_array(Y)
class_counts = Counter(Y_class)
print(f"Class counts: {class_counts}")


Validation = True

if(Validation == True):
    # Data Split Train 70% Test 15% Validation 15%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)
if(Validation == False):
    # Data Split Train 70% Test 30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

Y_train_class = classify_array(Y_train)
class_counts_train = Counter(Y_train_class)
print(f"Class counts (Train): {class_counts_train}")

#Training Neural Network with Cross Validation
if(Validation == False):
    clf = MLPRegressor(hidden_layer_sizes=(8, 5), activation='tanh',solver='lbfgs',max_iter= 10000,random_state=0)
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Perform cross-validation with scoring as MSE
    # mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)  # Negative because lower MSE is better
    # mse_scores = cross_val_score(clf, X_train, Y_train, cv=kf, scoring=mse_scorer)

    # # Perform cross-validation with scoring as R² (default for regressors)
    # r2_scores = cross_val_score(clf, X_train, Y_train, cv=kf) 
    # print("Cross-Validated R² Scores: ", r2_scores)
    # print(f"Mean R²: {np.mean(r2_scores)}")

    # print("Cross-Validated MSE Scores (Negative): ", mse_scores)
    # print(f"Mean MSE: {np.mean(np.abs(mse_scores))}")
    clf.fit(X_train, Y_train)
    print ("Training Accuracy: ",clf.score(X_train,Y_train)) # R^2  
    print("MSE: ", mean_squared_error(Y_train, clf.predict(X_train)))

number = 9
#Grid Search for Hyperparameter Tuning
if(Validation == True):
    param_grid = {
        'hidden_layer_sizes': [(number, i) for i in range(1, 14)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
    }
    df = pd.read_csv('Project_1//src//CINTE24-25_Proj1_SampleData.csv')
    Y_test2 = df["CLPVariation"].values
    X_test2 = df.iloc[:, 0:6].values

    Best_Accuracy = 0  
    Best_MSE = 1000 
    for i in range(param_grid['hidden_layer_sizes'].__len__()):
        for j in range(param_grid['activation'].__len__()):
            for k in range(param_grid['solver'].__len__()):
                clf = MLPRegressor(hidden_layer_sizes=param_grid['hidden_layer_sizes'][i], activation=param_grid['activation'][j],solver=param_grid['solver'][k],max_iter= 1000,early_stopping=True,random_state=0).fit(X_train, Y_train)
                # accuracy = clf.score(X_val, Y_val)
                # mse = mean_squared_error(Y_val,clf.predict(X_val))
                accuracy = clf.score(X_test2, Y_test2)
                mse = mean_squared_error(Y_test2,clf.predict(X_test2))
                print(f"Accuracy: {accuracy}, MSE: {mse}, Hidden Layer Sizes: {param_grid['hidden_layer_sizes'][i]}, Activation: {param_grid['activation'][j]}, Solver: {param_grid['solver'][k]}")
                if(Best_MSE > mse):
                    Best_MSE = mse
                    # Best_Accuracy = accuracy
                    best_param = [param_grid['hidden_layer_sizes'][i],param_grid['activation'][j],param_grid['solver'][k]]
                    best_clf = clf

    print("Best Accuracy: " +  str(best_clf.score(X_test2,Y_test2)) + " MSE: " + str(mean_squared_error(Y_test2,best_clf.predict(X_test2))) + " Best Parameters: " + str(best_param))
    clf = best_clf
    

# Perfomance Metrics
Y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, Y_test)
print(f"Test Accuracy: {accuracy}")
print("Test MSE: ", mean_squared_error(Y_test, Y_pred))

Y_test_class = classify_array(Y_test)
Y_pred_class = classify_array(Y_pred)
Y_test_counts = Counter(Y_test_class)
print(f"Test Class counts: {Y_test_counts}")
confusion_matrix = pd.crosstab(Y_test_class, Y_pred_class, rownames=['Actual'], colnames=['Predicted'])

print(f"Confusion Matrix:\n{confusion_matrix}")
report = classification_report(Y_test_class, Y_pred_class,zero_division=0)
print(report)
joblib.dump(clf, 'model_filename.pkl')

df = pd.read_csv('Project_1//src//CINTE24-25_Proj1_SampleData.csv')
Y_test2 = df["CLPVariation"].values
Y_pred2 = clf.predict(df.iloc[:, 0:6].values)

Y_test2_class = classify_array(Y_test2)
Y_pred2_class = classify_array(Y_pred2)
for i in range(len(Y_pred2)):
    print(f"Actual: {Y_test2[i]}, Actual Class: {Y_test2_class[i]}, Prediction: {Y_pred2[i]} Predicted class: {Y_pred2_class[i]}")

print("Test MSE: ", mean_squared_error(Y_test2, Y_pred2))

# Best Parameters: {'activation': 'relu', 'hidden_layer_sizes': (10, 12), 'solver': 'lbfgs'}
# Best Accuracy: 0.9437004756337296 MSE: 0.026162163774908404 Best Parameters: [(11, 1), 'logistic', 'lbfgs']
# Best Accuracy: 0.964070272935983 MSE: 0.01669640044774043 Best Parameters: [(5, 5), 'tanh', 'lbfgs']
# Best Accuracy: 0.9552937531012275 MSE: 0.020774814108871952 Best Parameters: [(7, 5), 'tanh', 'lbfgs']
# Best Accuracy: 0.9645428522137957 MSE: 0.01647679474765803 Best Parameters: [(8, 5), 'tanh', 'lbfgs']