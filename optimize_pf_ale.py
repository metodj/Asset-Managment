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


def read_batch_multi_seq2seq_window_back(X, window=10, window_back=50):
    n = X.shape[0] - window - window_back + 1
    inputs_encoder = np.zeros((n, window_back, X.shape[1]))
    inputs_decoder = np.zeros((n, window, X.shape[1]))
    targets = np.zeros((n, window, X.shape[1]))
    for i in range(n):
        inputs_encoder[i, :, :] = X.iloc[i:i + window_back, :].values
        inputs_decoder[i, :, :] = X.iloc[window_back - 1 + i:window_back - 1 + i + window, :].values
        targets[i, :, :] = X.iloc[window_back + i:window_back + i + window, :].values
    return inputs_encoder, inputs_decoder, targets


def RNNLSTM(X, rebalancing_dates_counter, model_path):

    future = 20
    nr_taps = 10
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
        if epoch_error < 1e-5:  # form of early stopping
            break
    model.save_model(model_path)
    last_x = np.zeros((1, 1, nr_taps * x_s))
    for i in range(nr_taps):
        last_x[0, 0, i*x_s:(i+1)*x_s] = X.iloc[-nr_taps+i, :]

    out = model.predict_batch(last_x)

    return out[0].ravel()


def RNNLSTM_seq2seq(X, window, window_back, rebalancing_dates_counter, model_path):
    def generate_batch():
        x, y, z = read_batch_multi_seq2seq_window_back(X)
        return x, y, z

    hidden_nodes = 20
    x_s = X.shape[1]

    batch_size = X.shape[0] - window - window_back + 1
    print("BATCH SIZE:")
    print(batch_size)

    model = ls.Model_Seq2seq(input_size=x_s, output_size=x_s, batch_size=batch_size, rnn_hidden=hidden_nodes)
    model.build()
    model.restore_model(rebalancing_dates_counter, model_path)
    if rebalancing_dates_counter % 5 == 0:
        for epoch in range(5):
            epoch_error = model.train_batch(generate_batch=generate_batch)
            print(epoch_error)

        model.save_model(model_path)
        last_x = X.iloc[-window_back:, :].values.reshape((1, window_back, x_s))
        out = model.predict_batch(last_x, window)
    else:
        last_x = X.iloc[-window_back:, :].values.reshape((1, window_back, x_s))
        out = model.predict_batch(last_x, window)
    print(out)
    return out


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


def optimize(x, ra, method=None,  window=0, window_back=0, rebalancing_dates_counter=0):

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
        model_path = "./model_lstm.ckpt"
        ret = RNNLSTM(x, rebalancing_dates_counter, model_path)

    if method is 'LSTM_seq2seq':
        model_path = "model_seq2seq.ckpt"
        ret = RNNLSTM_seq2seq(x, window, window_back, rebalancing_dates_counter, model_path)

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

    #Annualized return
    pnl_shape = pnl.cumsum().shape[0]
    ann_return = (1 + pnl.cumsum().iloc[-1]) ** (n / pnl_shape) - 1

    return {"sharpe": sharpe, "return": ret, "mdd": mdd, "ann_return": ann_return}

def run_pipeline_lstm(method, risk_aversion, window, window_back, start_investing_period):
    # set dates (and freq)
    dtindex = pd.bdate_range('1992-12-31', '2015-12-28', weekmask='Fri', freq='C')
    start_investing_period = dtindex.get_loc(start_investing_period)

    rebalancing_period = window

    rebalancing_dates = dtindex[start_investing_period::rebalancing_period]  # to warm-up LSTM
    print("Start: ", rebalancing_dates[0])
    df = pd.read_csv('markets_new.csv', delimiter=',')
    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))
    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])

    input_returns = df0.pct_change().fillna(0)
    input_returns = input_returns.iloc[1:, :]
    weights = pd.DataFrame(data=np.nan, columns=input_returns.columns, index=input_returns.index)

    # param for restoring the LSTM model
    rebalancing_dates_counter = 0

    for date in dtindex[window - 1:]:
        today = date
        if rebalancing_dates_counter == 0:
            returns = input_returns.loc[:today, :]  # .tail(window)
        else:
            if method == 'LSTM_seq2seq':  # for seq2seq we need longer intermediate periods
                returns = input_returns.loc[:today, :].tail(2 * window_back)
            else:
                returns = input_returns.loc[:today, :].tail(window)
        last = returns.index[-2]

        if today in rebalancing_dates:  # re-optimize and get new weights
            print("rebalancing date: ", today)
            weights.loc[today, :] = optimize(returns, risk_aversion, method, window, window_back, rebalancing_dates_counter)
            rebalancing_dates_counter += 1
        else:  # no re-optimization, re-balance the weights
            weights.loc[today, :] = weights.loc[last, :] * (1 + returns.loc[today, :]) \
                                    / (1 + (weights.loc[last, :] * returns.loc[today, :]).sum())

    pnl = (weights.shift(1) * input_returns).sum(axis=1)

    # Max-Drawdown calculation
    md = pnl.cumsum()[start_investing_period:]
    Roll_Max = md.rolling(window=md.shape[0], min_periods=1).max()
    # Daily_Drawdown = md / Roll_Max - 1.0
    Daily_Drawdown = md - Roll_Max
    mdd = -Daily_Drawdown.min()

    plt.figure
    pnl.cumsum().plot()

    # Annualized return
    pnl_shape = pnl.cumsum()[start_investing_period:].shape[0]
    # change to 252 if we have daily data!!
    ann_return = (1 + pnl.cumsum().iloc[-1]) ** (52 / pnl_shape) - 1

    # sharpe
    sharpe = pnl.mean() / pnl.std() * np.sqrt(52)

    #cum return
    ret = pnl.cumsum().iloc[-1]

    plt.title('Sharpe : {:.3f} \n Total return: {:.3f}, Annunalized return: {:.3f} \n Max drawdown: {:.3f}'. \
              format(sharpe, pnl.cumsum().iloc[-1], ann_return, abs(Daily_Drawdown.min())))

    plt.figure
    df0.pct_change().cumsum().plot()

    plt.figure
    weights.plot()

    plt.show()

    return {"sharpe": sharpe, "return": ret, "mdd": mdd, "ann_return": ann_return}


if __name__ == '__main__':

    #PIPELINE SETTINGS
    method = 'LSTM_Multi'
    risk_aversion = 1

    #Note: window and rebalancing_period always expressed in weeks
    window = 52
    rebalancing_period = 52

    start_date = '2004-12-31'
    end_date = '2015-12-28'
    weekmask = True

    if weekmask:
        dtindex = pd.bdate_range(start_date, end_date, weekmask='Fri', freq='C')
    else:
        dtindex = pd.bdate_range(start_date, end_date, freq='C')
        window = window * 5
        rebalancing_period = rebalancing_period * 5

    if "LSTM" in method:
        start_investing_period = '2004-12-31'
        window_back = 50
        results = run_pipeline_lstm(method, risk_aversion, window, window_back, start_investing_period)
    else:
        results = run_pipeline(method, risk_aversion, window, rebalancing_period, dtindex, weekmask)


    print('\n============ RESULTS ============')
    print('\nMethod: {}\nSharpe Ratio: {:.3f}\nMax DD: {:.3f}\nTotal return: {:.3f}\nAnnualized return:{:.3f}'. \
          format(method, results['sharpe'], results['mdd'], results['return'], results['ann_return']))