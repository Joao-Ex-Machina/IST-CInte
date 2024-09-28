
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
from scipy import stats

df = pd.read_csv('DCOILBRENTEUv2.csv')
df['Var']=df['DCOILBRENTEU']-df.shift(periods=1, axis="index")['DCOILBRENTEU']
plt.hist(df['Var'])
plt.show()
print(df)

