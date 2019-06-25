import optimize_pf_ale as opt
import numpy as np
import pandas as pd
import HMM as hmm
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

def plot_states():
    dtindex = pd.bdate_range('2007-12-31', '2009-12-28', freq='C')
    df = pd.read_csv('GSPC.csv', delimiter=',')

    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%Y-%m-%d'))
    df0 = pd.DataFrame(df0['Close']).rename(columns={"Close": "GSPC"})

    for filename in ['FVX', 'VIX', "GDAXI"]:
        df1 = pd.read_csv(filename + '.csv', delimiter=',')

        df1 = pd.DataFrame(data=df1.values, columns=df1.columns,
                           index=pd.to_datetime(df1['Date'], format='%Y-%m-%d'))
        df1 = pd.DataFrame(df1['Close']).rename(columns={"Close": filename})
        df0 = df1.join(df0, on='Date')

    df0 = df0.reindex(dtindex)
    # df0 = df0.drop(columns=['Date'])
    print(df0)
    # df0.dropna(axis=0, inplace=True)
    df0 = df0.pct_change().fillna(0)
    # print(df0)
    data = df0
    print(data)
    # data.dropna(inplace=True)

    K = 3  # number of clusters
    iter = 30  # number of epochs (EM algo)

    posteriori_prob, mu_s, cov_s, pred, states = hmm.expectation_maximization_mod(data, data, K, iter=iter)

    # regimes
    plt.plot(posteriori_prob.T)

    # annotated underlyings
    maxInd = np.argmax(posteriori_prob, axis=0)
    dax = data.iloc[:, 0].cumsum()
    vix = data.iloc[:, 1].cumsum()
    fvx = data.iloc[:, 2].cumsum()
    sp = data.iloc[:, 3].cumsum()
    plt.figure()
    for i in range(0, K):
        ndax = np.nan * dax
        nvix = np.nan * vix
        nfvx = np.nan * fvx
        nsp = np.nan * sp
        nsp[maxInd == i] = sp[maxInd == i]
        ndax[maxInd == i] = dax[maxInd == i]
        nfvx[maxInd == i] = fvx[maxInd == i]
        nvix[maxInd == i] = vix[maxInd == i]
        plt.subplot(411), plt.plot(nsp), plt.title('SPX')
        plt.subplot(412), plt.plot(ndax), plt.title('DAX')
        plt.subplot(413), plt.plot(nfvx), plt.title('FVX')
        plt.subplot(414), plt.plot(nvix), plt.title('VIX')

    g = pd.DataFrame(posteriori_prob.T, columns=range(0, K))
    g.to_csv('regimes.csv')
    plt.show()

def plot_wrong_states():
    dtindex = pd.bdate_range('2007-12-31', '2009-12-28', freq='C')

    df = pd.read_csv('markets_new.csv', delimiter=',')

    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))

    df0 = df0.reindex(dtindex)
    df0 = df0.drop(columns=['Date'])
    data = df0.pct_change().fillna(0)

    K = 3  # number of clusters
    iter = 30  # number of epochs (EM algo)

    print(data)

    posteriori_prob, mu_s, cov_s, pred = hmm.expectation_maximization(data, K, iter=iter)

    # regimes
    plt.plot(posteriori_prob.T)

    # annotated underlyings
    maxInd = np.argmax(posteriori_prob, axis=0)
    dax = data.iloc[:, 0].cumsum()
    vix = data.iloc[:, 1].cumsum()
    fvx = data.iloc[:, 2].cumsum()
    sp = data.iloc[:, 3].cumsum()
    plt.figure()
    for i in range(0, K):
        ndax = np.nan * dax
        nvix = np.nan * vix
        nfvx = np.nan * fvx
        nsp = np.nan * sp
        nsp[maxInd == i] = sp[maxInd == i]
        ndax[maxInd == i] = dax[maxInd == i]
        nfvx[maxInd == i] = fvx[maxInd == i]
        nvix[maxInd == i] = vix[maxInd == i]
        plt.subplot(411), plt.plot(nsp), plt.title('SPX')
        plt.subplot(412), plt.plot(ndax), plt.title('DAX')
        plt.subplot(413), plt.plot(nfvx), plt.title('FVX')
        plt.subplot(414), plt.plot(nvix), plt.title('VIX')

    g = pd.DataFrame(posteriori_prob.T, columns=range(0, K))
    g.to_csv('regimes.csv')
    plt.show()


#plot_states()
plot_wrong_states()
