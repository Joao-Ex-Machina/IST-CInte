import subprocess
import joblib
import pandas as pd
import numpy as np
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
        if value < -1:
            return "Decrease"
        if value > 1:
            return "Increase"
    print("Value: ", value)
    return "Out of Range"

# Function to classify an entire array
def classify_array(array):
    return [classify_value(val) for val in array]

#DEFAULT="tests/CINTE24-25_Proj1_SampleData.csv"
DEFAULT="Proj1_TestS.csv"

result = subprocess.run(["python", "Fuzzy_System.py", DEFAULT], capture_output=True, text=True)

print(result.stdout)

data = pd.read_csv(DEFAULT)
clf = joblib.load('model_filename.pkl')

X = data.iloc[:, 0:6].values

Y_pred = clf.predict(X)
Y_pred_class = classify_array(Y_pred)

Fuzz_Res = pd.read_csv("TestResult_FISOnly.csv")

Fuzzy_class = classify_array(Fuzz_Res['FuzzyCLPVar'])

#Fuzz_Res = pd.DataFrame(Fuzz_Res, columns=["FuzzyCLPVar"])
Results = pd.DataFrame({'FuzzyCLP':Fuzz_Res["FuzzyCLPVar"],'FuzzyClass':Fuzzy_class,'NNCLPVar':Y_pred, 'NNClass':Y_pred_class})

Results.to_csv("TestResult.csv")
