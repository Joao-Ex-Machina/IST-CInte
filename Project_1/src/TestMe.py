import subprocess
import joblib
import pandas as pd

DEFAULT_debug="CINTE24-25_Proj1_SampleData.csv"
DEFAULT="Proj1_TestS.csv"

result = subprocess.run(["python", "Fuzzy_System.py", DEFAULT_debug], capture_output=True, text=True)

print(result.stdout)

data = pd.read_csv(DEFAULT)
clf = joblib.load('model_filename.pkl')

X = data.iloc[:, 0:6].values

Y_pred = clf.predict(X)
Y_pred = pd.DataFrame(Y_pred, columns=["NN_CLPVariation"])

