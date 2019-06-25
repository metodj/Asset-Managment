import os
import pandas as pd
import osqp
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance as cv
import HMM as hmm
import LSTM as ls
from hmmlearn import hmm as hmml


def read_batch_multi(X, batch_size, future=10, nr_taps=2, batch_start=0):
    num_features = 1
    s = X
    x = np.zeros((num_features, batch_size, s.shape[1]*nr_taps))
    y = np.zeros((num_features, batch_size, s.shape[1]))
    for i in range(nr_taps):
        x[0, :, i * s.shape[1]:(i + 1) * s.shape[1]] = s.iloc[i:-future - nr_taps + i + 1, :]
    y[0, :, :] = s.rolling(future).mean().iloc[future+nr_taps-1:, :]
    return x, y

def RNNLSTM(X):

    def generate_batch(N):
        assert(X.shape[0] > N)
        x, y = read_batch_multi(X, N, future=future, nr_taps=nr_taps)
        ys = 0.1*y + 0.9*y.mean(axis=2)[:, :, np.newaxis]       # shrinkage

        return x, ys

    future = 25
    nr_taps = 5
    hidden_nodes = 15
    x_s = X.shape[1]     # columns

    batch_size = X.shape[0] - future - nr_taps + 1
    print("BATCH SIZE:")
    print(batch_size)

    model = ls.Model(input_size=nr_taps*x_s, output_size=x_s, rnn_hidden=hidden_nodes)
    model.build()
    for epoch in range(10):
        epoch_error = model.train_batch(generate_batch=generate_batch, batch_size=X.shape[0]-future-nr_taps+1)
        print(epoch_error)
    last_x = np.zeros((1, 1, nr_taps * x_s))
    for i in range(nr_taps):
        last_x[0, 0, i*x_s:(i+1)*x_s] = X.iloc[-nr_taps+i, :]

    out = model.predict_batch(last_x)

    return out[0].ravel()


def HMM(X):
    K = 3
    p = .1
    iter = 20
    posteriori_prob, mu_s, cov_s, pred = hmm.expectation_maximization(X, K, iter, p)

    return pred

def HMM_mod(X):
    K = 3
    p = .1
    iter = 20
    posteriori_prob, mu_s, cov_s, pred, states = hmm.expectation_maximization_mod(X[['VIX', 'FVX', 'GSPC', 'GDAXI']], X.drop(columns=['VIX', 'FVX', 'GSPC', 'GDAXI']), K, iter, p)

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


def optimize(x, ra, method=None):

    ret = x.mean().fillna(0).values

    if method is 'js_mean':
        ret = js_mean(x)

    if method is 'HMM':
        ret = HMM(x)

    if method is 'by_mean':
        ret = by_mean(x)

    if method is 'HMM_mod':
        ret = HMM_mod(x)
        x.drop(columns=['VIX', 'FVX', 'GSPC', 'GDAXI'], inplace=True)

    if method is 'LSTM_Multi':
        ret = RNNLSTM(x)

    # one of the baselines
    if method is 'equal_weights':
        return (1/x.shape[1]) * np.ones(x.shape[1])

    cov = ra * pd.DataFrame(data=cv.oas(x)[0], index=x.columns, columns=x.columns).fillna(0)

    #Second baseline
    if method is 'Sample_mean':
        ret = x.mean().fillna(0).values
        cov = ra * pd.DataFrame(data=pd.DataFrame(x).cov(), index=x.columns, columns=x.columns).fillna(0)



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

    problem.setup(sCov, -ret, sA, l, u, verbose=False)

    # Solve problem
    res = problem.solve()
    pr = pd.Series(data=res.x, index=x.columns)
    return pr


def run_pipeline(method, risk_aversion, window, rebalancing_period, dtindex, weekmask):
    print('Running pipeline for {}'.format(method))
    # set dates (and freq)
    rebalancing_dates = dtindex[window - 1::rebalancing_period]

    df = pd.read_csv('markets_new.csv', delimiter=',')

    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))

    if ("expanded" in method) or ("mod" in method):
        # Adding indices to the data frame
        print('Adding indices to the data frame')

        for filename in ['VIX', 'FVX', 'GSPC', 'GDAXI']:
            df1 = pd.read_csv(filename + '.csv', delimiter=',')

            df1 = pd.DataFrame(data=df1.values, columns=df1.columns, index=pd.to_datetime(df1['Date'], format='%Y-%m-%d'))
            df1 = pd.DataFrame(df1['Close']).rename(columns={"Close": filename})
            df0 = df1.join(df0, on='Date')

    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])

    if ("expanded" in method) or ("mod" in method):
        df_extended = df0.copy()
        df0.drop(columns=['VIX', 'FVX', 'GSPC', 'GDAXI'], inplace=True)
        input_returns_extended = df_extended.pct_change().fillna(0)

    input_returns = df0.pct_change().fillna(0)

    input_returns = input_returns.iloc[1:, :]
    weights = pd.DataFrame(data=np.nan, columns=input_returns.columns, index=input_returns.index)
    for date in dtindex[window - 1:]:
        today = date
        returns = input_returns.loc[:today, :].tail(window)
        last = returns.index[-2]

        if today in rebalancing_dates:  # re-optimize and get new weights
            print('{}: Rebalancing weights'.format(today))
            if ("expanded" in method) or ("mod" in method):
                returns_ext = input_returns_extended.loc[:today, :].tail(window)
                weights.loc[today, :] = optimize(returns_ext, risk_aversion, method)
            else:
                weights.loc[today, :] = optimize(returns, risk_aversion, method)
        else:  # no re-optimization, re-balance the weights
            print('{}: relocate assets to preserve wights'.format(today))
            weights.loc[today, :] = weights.loc[last, :] * (1 + returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * returns.loc[today, :]).sum())

    pnl = (weights.shift(1) * input_returns).sum(axis=1)

    #Indicators calculation
    if weekmask:
        n = 52
    else:
        n = 252

    # Max drawdown
    md = pnl.cumsum()[window:]
    Roll_Max = md.rolling(window=md.shape[0], min_periods=1).max()
    Daily_Drawdown = md - Roll_Max
    mdd = -Daily_Drawdown.min()

    #Sharpe
    sharpe = pnl.mean() / pnl.std() * np.sqrt(n)

    #Return
    ret = pnl.cumsum().iloc[-1]

    #Sortino
    target = 0.05
    df = pd.DataFrame(data=pnl)
    df['downside_returns'] = 0
    df.loc[pnl < target, 'downside_returns'] = pnl ** 2
    expected_return = pnl.mean()
    down_stdev = np.sqrt(df['downside_returns'].mean())
    sortino_ratio = expected_return / down_stdev * np.sqrt(n)

    #Annualized return
    pnl_shape = pnl.cumsum().shape[0]
    ann_return = (1 + pnl.cumsum().iloc[-1]) ** (n / pnl_shape) - 1

    return {"sharpe": sharpe, "sortino": sortino_ratio, "return": ret, "mdd": mdd, "ann_return": ann_return}

if __name__ == '__main__':

    #PIPELINE SETTINGS
    method = 'HMM_mod'
    risk_aversion = 1

    #Note: window and rebalancing_period always expressed in weeks
    window = 52
    rebalancing_period = 52

    start_date = '2012-12-31'
    end_date = '2015-12-28'
    weekmask = True

    if weekmask:
        dtindex = pd.bdate_range(start_date, end_date, weekmask='Fri', freq='C')
    else:
        dtindex = pd.bdate_range(start_date, end_date, freq='C')
        window = window * 5
        rebalancing_period = rebalancing_period * 5

    results = run_pipeline(method, risk_aversion, window, rebalancing_period, dtindex, weekmask)

    print('============ RESULTS ============')
    print('\nMethod: {}\nSharpe Ratio: {:.3f}\nSortino Ratio: {:.3f}\nMax DD: {:.3f}\nTotal return: {:.3f}\nAnnualized return:{:.3f}'. \
          format(method, results['sharpe'], results['sortino'], results['mdd'], results['return'], results['ann_return']))
