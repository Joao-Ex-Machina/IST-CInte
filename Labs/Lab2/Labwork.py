
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('DCOILBRENTEUv2.csv')

#df1 = df.sort_values(by=['Date'])
print(df)
minim =  df['DCOILBRENTEU'].min()
maxim = df['DCOILBRENTEU'].max()
df['Norm']=df['DCOILBRENTEU']-minim/(maxim-minim)
df['Stan']=(df['DCOILBRENTEU']-df['DCOILBRENTEU'].mean())/df['DCOILBRENTEU'].std()
df['Norm-rolling']=df['Norm'].rolling(50).mean()
plt.plot(df["Norm"])
plt.plot(df["Stan"])
plt.plot(df['Norm-rolling'])
plt.show()

df.plot('DATE', 'Norm-rolling')
plt.show()
print(df)

