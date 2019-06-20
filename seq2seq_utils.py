import pandas as pd
import numpy as np

#### data preprocessing
dtindex = pd.bdate_range('1992-12-31', '2015-12-28', weekmask='Fri', freq='C')
df = pd.read_csv('markets_new.csv', delimiter=',')
df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))
df0 = df0.reindex(dtindex)
df0 = df0.drop(columns=['Date'])
df0 = df0.fillna(0)

train = df0.iloc[:1000, :]
eval_ = df0.iloc[1000:, :]
WINDOW = 20

def df_to_numpy(df, window):
    n = df.shape[0]//window
    X = np.zeros(shape=(n, window, df.shape[1]))
    for i in range(n):
        X[i, :, :] = df.iloc[i*window:window*(i+1), :]
    return X


train_np = df_to_numpy(train, WINDOW)
eval_np = df_to_numpy(eval_, WINDOW)

np.save("train.npy", train_np)
np.save("eval.npy", eval_np)