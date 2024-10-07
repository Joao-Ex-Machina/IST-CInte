import subprocess
import joblib

DEFAULT_debug="CINTE24-25_Proj1_SampleData.csv"
DEFAULT="Proj1_TestS.csv"

result = subprocess.run(["python", "Fuzzy_System.py", DEFAULT_debug], capture_output=True, text=True)

print(result.stdout)

clf = joblib.load('model_filename.pkl')

