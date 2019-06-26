import os
import pandas as pd
import osqp
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance as cv
import HMM as hmm
import LSTM as ls


def read_batch_multi(X, batch_size, future=10, nr_taps=2, batch_start=0):
    num_features = 1
    s = X
    x = np.zeros((num_features, batch_size, s.shape[1]*nr_taps))
    y = np.zeros((num_features, batch_size, s.shape[1]))
    for i in range(nr_taps):
        x[0, :, i * s.shape[1]:(i + 1) * s.shape[1]] = s.iloc[i:-future - nr_taps + i + 1, :]
    y[0, :, :] = s.rolling(future).mean().iloc[future+nr_taps-1:, :]
    return x, y

def RNNLSTM(X, rebalancing_dates_counter, model_path):

    future = 25
    nr_taps = 5
    hidden_nodes = 15
    x_s = X.shape[1]  # columns

    def generate_batch(N):
        assert(X.shape[0] > N)
        x, y = read_batch_multi(X, N, future=future, nr_taps=nr_taps)
        ys = 0.1*y + 0.9*y.mean(axis=2)[:, :, np.newaxis]  # shrinkage

        return x, ys

    batch_size = X.shape[0] - future - nr_taps + 1
    print("BATCH SIZE:")
    print(batch_size)

    model = ls.Model(input_size=nr_taps*x_s, output_size=x_s, rnn_hidden=hidden_nodes)
    model.build()
    model.restore_model(rebalancing_dates_counter, model_path)
    for epoch in range(10):
        epoch_error = model.train_batch(generate_batch=generate_batch, batch_size=X.shape[0]-future-nr_taps+1)
        print(epoch_error)
        if epoch_error < 1e-5:
            break
    model.save_model(model_path)
    last_x = np.zeros((1, 1, nr_taps * x_s))
    for i in range(nr_taps):
        last_x[0, 0, i*x_s:(i+1)*x_s] = X.iloc[-nr_taps+i, :]

    out = model.predict_batch(last_x)

    return out[0].ravel()


def HMM(X):
    K = 4
    p = .1
    iter = 40
    posteriori_prob, mu_s, cov_s, pred = hmm.expectation_maximization(X, K, iter, p)

    return pred


def js_mean(X):
    sigma = 5*np.max(X.std())
    dim = X.shape[1]
    assert(dim > 2)
    norm_x = np.linalg.norm(X.mean())

    js_factor = np.max((1-(sigma**2 * (dim - 2)/norm_x), 0.01))
    js_estimator = js_factor * X.mean()

    return js_estimator.values


def by_mean(X):
    Factor = .01
    sigma = np.max(X.std())
    sigma_0 = Factor * sigma
    m = X.shape[0]
    mu_0 = np.mean(X.mean()/X.std())
    return ((np.sum(X, axis=0)/sigma**2 + mu_0/sigma_0**2)/(m/sigma**2 + 1/sigma_0**2)).values


def optimize(x, ra, rebalancing_dates_counter, model_path, method=None):

    ret = x.mean().fillna(0).values

    cov = ra * pd.DataFrame(data=cv.oas(x)[0],index=x.columns, columns=x.columns).fillna(0)

    if method is 'js_mean':
        ret = js_mean(x)

    if method is 'HMM':
        ret = HMM(x)

    if method is 'by_mean':
        ret = by_mean(x)

    if method is 'LSTM_Multi':
        ret = RNNLSTM(x, rebalancing_dates_counter, model_path)

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
    method = 'equal_weights'
    risk_aversion = 1
    window = 150
    warm_up_rnn = 4 #number of windows, for rnn to warmup

    # set dates (and freq)
    dtindex = pd.bdate_range('1992-12-31', '2015-12-28', weekmask='Fri', freq='C')
    rebalancing_period = window

    rebalancing_dates = dtindex[warm_up_rnn*window-1::rebalancing_period] #to warm-up LSTM
    # print(rebalancing_dates)
    df = pd.read_csv('markets_new.csv', delimiter=',')
    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))
    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])

    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]
    weights = pd.DataFrame(data=np.nan, columns=input_returns.columns, index=input_returns.index)

    # param for restoring the LSTM model
    rebalancing_dates_counter = 0
    model_path = "model_lstm.ckpt"

    for date in dtindex[window - 1:]:
        today = date
        returns = input_returns.loc[:today, :] # .tail(window)
        last = returns.index[-2]

        if today in rebalancing_dates:  # re-optimize and get new weights
            print("rebalancing date: ", today)
            weights.loc[today, :] = optimize(returns, risk_aversion, rebalancing_dates_counter, model_path, method)
            rebalancing_dates_counter += 1
        else:  # no re-optimization, re-balance the weights
            weights.loc[today, :] = weights.loc[last, :] * (1 + returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * returns.loc[today, :]).sum())

    pnl = (weights.shift(1)*input_returns).sum(axis=1)


    # Max-Drawdown calculation
    md = pnl.cumsum()[warm_up_rnn*window:]
    Roll_Max = md.rolling(window=md.shape[0], min_periods=1).max()
    # Daily_Drawdown = md / Roll_Max - 1.0
    Daily_Drawdown = md - Roll_Max



    plt.figure
    pnl.cumsum().plot()

    # Annualized return
    pnl_shape = pnl.cumsum()[warm_up_rnn*window:].shape[0]
    # change to 252 if we have daily data!!
    ann_return = (1 + pnl.cumsum().iloc[-1])**(52/pnl_shape) - 1

    plt.title('Sharpe : {:.3f} \n Total return: {:.3f}, Annunalized return: {:.3f} \n Max drawdown: {:.3f}'. \
              format(pnl.mean()/pnl.std()*np.sqrt(52), pnl.cumsum().iloc[-1], ann_return, abs(Daily_Drawdown.min())))

    plt.figure
    df0.pct_change().cumsum().plot()

    plt.figure
    weights.plot()

    plt.show()
