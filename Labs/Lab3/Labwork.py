
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
from scipy import stats

df = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', sep=';', decimal=',')
cols = df.select_dtypes('number').columns
df_sub = df.loc[:, cols]
df['Time']=pd.to_datetime(df['Time (UTC)'])
startTime=datetime.datetime.now()
zScore=np.abs(stats.zscore(df['Close']))
outlier_indices = np.where(zScore > 3)[0]
df_filtered = df.drop(outlier_indices)
plt.plot(df_filtered['Close'])
plt.show()
print(df)

