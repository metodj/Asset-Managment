import optimize_pf_ale as opt
import numpy as np
import pandas as pd
import HMM as hmm
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

def plot_states():
    dtindex = pd.bdate_range('2008-12-31', '2010-12-28', freq='C')
    df = pd.read_csv('VIX.csv', delimiter=',')

    df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%Y-%m-%d'))
    df0 = pd.DataFrame(df0['Close']).rename(columns={"Close": "VIX"})

    for filename in ['FVX', 'GSPC', 'GDAXI']:
        df1 = pd.read_csv(filename + '.csv', delimiter=',')

        df1 = pd.DataFrame(data=df1.values, columns=df1.columns,
                           index=pd.to_datetime(df1['Date'], format='%Y-%m-%d'))
        df1 = pd.DataFrame(df1['Close']).rename(columns={"Close": filename})
        df0 = df1.join(df0, on='Date')

    df0 = df0.reindex(dtindex)
    #df0 = df0.drop(columns=['Date'])
    df0 = df0.pct_change().fillna(0)
    print(df0)

    K = 2
    iter = 500
    p = 0.4
    posteriori_prob, mu_s, cov_s, pred, states = hmm.expectation_maximization_mod(df0, df0, K, iter, p)

    #model = hmml.GaussianHMM(n_components=K, n_iter=iter)

    print(states.shape, df0.shape)

    for i in range(K):
        print(len(states[states == i]))

    fig, axs = plt.subplots(
        K,
        sharex=True, sharey=True
    )
    colours = cm.rainbow(
        np.linspace(0, 1, K)
    )

    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = states == i
        ax.plot_date(
            df0.index[mask],
            df0['FVX'][mask],
            ".", linestyle='none',
            c=colour
        )
        ax.set_title("Hidden State #%s" % i)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()


plot_states()

