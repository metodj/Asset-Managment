import os
import pandas as pd
import osqp
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance as cv
import HMM as hmm
import LSTM as ls

# window = 10
# dtindex = pd.bdate_range('1992-12-31', '2015-12-28', weekmask='Fri', freq='C')
# rebalancing_period = window
# rebalancing_dates = dtindex[20*window::rebalancing_period] # warmup period of 20 windows
# df = pd.read_csv('markets_new.csv', delimiter=',')
# df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))
# df0 = df0.reindex(dtindex)
# df0 = df0.drop(columns=['Date'])
#
# input_returns = df0.pct_change().fillna(0)
# input_returns = input_returns.iloc[1:, :]
#
# today = rebalancing_dates[0]
# returns = input_returns.loc[:today, :]
# print(returns.shape)


def read_batch_multi_seq2seq(X, window=10):
    n = X.shape[0] // window - 1
    inputs_encoder = X.iloc[:-window,:].values.reshape((n, window, X.shape[1]))
    inputs_decoder = X.iloc[window-1:-1, : ].values.reshape((n, window, X.shape[1]))
    targets = X.iloc[window:, :].values.reshape((n, window, X.shape[1]))
    return inputs_encoder, inputs_decoder, targets


def read_batch_multi_seq2seq_window_back(X, window=10, window_back=100):
    n = X.shape[0] // window - window
    inputs_encoder = np.zeros((n, window_back, X.shape[1]))
    inputs_decoder = np.zeros((n, window, X.shape[1]))
    targets = np.zeros((n, window, X.shape[1]))
    for i in range(n):
        inputs_encoder[i, :, :] = X.iloc[i*window:i*window+window_back, :].values
        inputs_decoder[i, :, :] = X.iloc[window_back -1 + i*window:window_back -1 + (i+1)*window, :].values
        targets[i, :, :] = X.iloc[window_back + i*window:window_back + (i+1)*window, :].values
    return inputs_encoder, inputs_decoder, targets


def RNNLSTM_seq2seq(X, restore, window, window_back, rebalancing_dates_counter):

    def generate_batch(window_back_=True):
        if window_back_:
            x, y, z = read_batch_multi_seq2seq_window_back(X)
        else:
            x, y, z = read_batch_multi_seq2seq(X)
        return x, y, z

    hidden_nodes = 20
    x_s = X.shape[1]

    batch_size = X.shape[0] // window - window
    print("BATCH SIZE:")
    print(batch_size)

    model = ls.Model_seq2seq(input_size=x_s, output_size=x_s, batch_size=batch_size, rnn_hidden=hidden_nodes)
    model.build()
    if not restore:
        for epoch in range(10):
            epoch_error = model.train_batch(rebalancing_dates_counter, "model.ckpt", generate_batch=generate_batch, )
            print(epoch_error)

        model.save_model("model.ckpt")
        last_x = X.iloc[-window_back:, :].values.reshape((1, window_back, x_s))
        out = model.predict_batch(last_x, window, "model.ckpt")
    else:
        last_x = X.iloc[-window_back:, :].values.reshape((1, window_back, x_s))
        out = model.predict_batch(last_x, window, "model.ckpt")
    return out


def optimize(x, ra, restore, window, window_back, rebalancing_dates_counter, method=None):
    print(method)

    ret = x.mean().fillna(0).values

    cov = ra * pd.DataFrame(data=cv.oas(x)[0],index=x.columns, columns=x.columns).fillna(0)

    if method is 'LSTM_seq2seq':
        ret = RNNLSTM_seq2seq(x, restore, window, window_back, rebalancing_dates_counter)

    # one of the baselines
    if method is 'equal_weights':
        return (1/x.shape[1]) * np.ones(x.shape[1])

    problem = osqp.OSQP()
    k = len(ret)

    # Setup workspace
    """
    setup(self, P=None, q=None, A=None, l=None, u=None, **settings):
            Setup OSQP solver problem of the form
            minimize     1/2 x' * P * x + q' * x
            subject to   l <= A * x <= u
    """
    A = np.concatenate((pd.np.ones((1, k)), np.eye(k)), axis=0)
    sA = sparse.csr_matrix(A)
    l = np.hstack([1, np.zeros(k)])
    u = np.ones(k + 1)
    sCov = sparse.csr_matrix(cov)

    problem.setup(sCov, -ret, sA, l, u)

    # Solve problem
    res = problem.solve()
    pr = pd.Series(data=res.x, index=x.columns)
    return pr


if __name__ == '__main__':

    # load data ###################################################################
    method = 'LSTM_seq2seq'
    risk_aversion = 1
    window = 10
    window_back = 100

    # set dates (and freq)
    dtindex = pd.bdate_range('1992-12-31', '2015-12-28', weekmask='Fri', freq='C')
    rebalancing_period = window
    rebalancing_dates = dtindex[100 * window::rebalancing_period]  # warmup period of 100 windows
    print(rebalancing_dates)
    df = pd.read_csv('markets_new.csv', delimiter=',')
    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))
    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])

    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]
    weights = pd.DataFrame(data=np.nan, columns=input_returns.columns, index=input_returns.index)

    rebalancing_dates_counter = 0
    restore = False # so that we train the model on the first rebalance date
    for date in dtindex[window - 1:]:
        today = date
        returns = input_returns.loc[:today, :] # .tail(window)
        last = returns.index[-2]

        if today in rebalancing_dates:  # re-optimize and get new weights
            print(today)
            weights.loc[today, :] = optimize(returns, risk_aversion, restore, window, window_back, rebalancing_dates_counter, method)
            rebalancing_dates_counter += 1
            # only retrain the model every 5th epoch
            if rebalancing_dates_counter % 5 == 0:
                restore = False
            else:
                restore = True
        else:  # no re-optimization, re-balance the weights
            weights.loc[today, :] = weights.loc[last, :] * (1 + returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * returns.loc[today, :]).sum())

    pnl = (weights.shift(1)*input_returns).sum(axis=1)

    plt.figure
    pnl.cumsum().plot()

    plt.title('Sharpe : {:.3f}'.format(pnl.mean()/pnl.std()*np.sqrt(52)))

    plt.figure
    df0.pct_change().cumsum().plot()

    plt.figure
    weights.plot()

    plt.show()