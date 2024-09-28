
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
from scipy import stats

dfuk = pd.read_csv('DCOILBRENTEUv2.csv')
dftexas= pd.read_csv('DCOILWTICOv2.csv')
plt.scatter(dfuk)
plt.scatter(dftexas)

plt.show()
print(df)

