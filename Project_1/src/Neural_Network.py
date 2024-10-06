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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)
if(Validation == False):
    # Data Split Train 70% Test 30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

Y_train_class = classify_array(Y_train)
class_counts_train = Counter(Y_train_class)
print(f"Class counts (Train): {class_counts_train}")

# smote = SMOTE(random_state=0)
# X_train_balanced, Y_train_balanced = smote.fit_resample(X_train, Y_train)
# print(f"Balanced class counts: {Counter(Y_train_balanced)}")

if(Validation == False):
    clf = MLPRegressor(hidden_layer_sizes=(18,5), activation='logistic',solver='lbfgs',max_iter= 10000,random_state=0)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Perform cross-validation with scoring as MSE
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)  # Negative because lower MSE is better
    mse_scores = cross_val_score(clf, X_train, Y_train, cv=kf, scoring=mse_scorer)

    # Perform cross-validation with scoring as R² (default for regressors)
    r2_scores = cross_val_score(clf, X_train, Y_train, cv=kf)

    
    
    print("Cross-Validated R² Scores: ", r2_scores)
    print(f"Mean R²: {np.mean(r2_scores)}")

    print("Cross-Validated MSE Scores (Negative): ", mse_scores)
    print(f"Mean MSE: {np.mean(np.abs(mse_scores))}")
    clf.fit(X_train, Y_train)
    # print ("Training Accuracy: ",clf.score(X_train,Y_train)) # R^2  
    # print("MSE: ", mean_squared_error(Y_train, clf.predict(X_train)))

if(Validation == True):
    param_grid = {
        'hidden_layer_sizes': [(18,5),(12,5),(12,6),(12,7),(18,6),(6,10),(7,10),(8,10)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],}
    
    mlp = MLPRegressor(max_iter=10000, random_state=0)
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring=mse_scorer, n_jobs=-1)

    # Fit the grid search to your data
    grid_search.fit(X_train, Y_train)

    # Best parameters and best score
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Negative MSE, so flip the sign

    print(f"Best Parameters: {best_params}")
    print(f"Best Mean Squared Error: {best_score}")

    # Train a final model with the best parameters
    clf = grid_search.best_estimator_
    
    # Best_Accuracy = 0   
    # for i in range(param_grid['hidden_layer_sizes'].__len__()):
    #     for j in range(param_grid['activation'].__len__()):
    #         for k in range(param_grid['solver'].__len__()):
    #             clf = MLPRegressor(hidden_layer_sizes=param_grid['hidden_layer_sizes'][i], activation=param_grid['activation'][j],solver=param_grid['solver'][k],max_iter= 1000,early_stopping=True,random_state=0).fit(X_train, Y_train)
    #             # clf = MLPClassifier(hidden_layer_sizes=param_grid['hidden_layer_sizes'][i], activation=param_grid['activation'][j],solver=param_grid['solver'][k],max_iter= 1000).fit(X_train_balanced, Y_train_balanced)
    #             accuracy = clf.score(X_val, Y_val)
    #             print(f"Accuracy: {accuracy}, Hidden Layer Sizes: {param_grid['hidden_layer_sizes'][i]}, Activation: {param_grid['activation'][j]}, Solver: {param_grid['solver'][k]}")
    #             if(accuracy > Best_Accuracy):
    #                 Best_Accuracy = accuracy
    #                 best_param = [param_grid['hidden_layer_sizes'][i],param_grid['activation'][j],param_grid['solver'][k]]

    # #Accuracy: 0.9586666666666667, Hidden Layer Sizes: (10, 4), Activation: relu, Solver: adam
    # print("Best Accuracy: " +  str(Best_Accuracy) + " MSE: " + str(mean_squared_error(Y_val,clf.predict(X_val))) + " Best Parameters: " + str(best_param))
    # clf = MLPRegressor(hidden_layer_sizes=best_param[0], activation=best_param[1],solver=best_param[2],max_iter= 1000,random_state=0).fit(X_train, Y_train) 
    # # clf = MLPClassifier(hidden_layer_sizes=best_param[0], activation=best_param[1],solver=best_param[2],max_iter= 1000,early_stopping=True).fit(X_train_balanced, Y_train_balanced)               


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
