import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import osqp
from scipy import sparse
from sklearn import covariance as cv
import sys

class HMMPortfolioOptimizer:
    def __init__(self, start_date, end_date, weekmask):
        #loads data
        if weekmask:
            dtindex = pd.bdate_range(start_date, end_date, weekmask='Fri', freq='C')
        else:
            dtindex = pd.bdate_range(start_date, end_date, freq='C')

        self.training_data = None
        self.weekmask = weekmask
        self.dtindex = dtindex
        self.K = 3
        self.iter = 20
        self.p = 0.1
        self.tradable_assets = None
        self.A = None
        self.g = None
        self.states = None

    def set_iterations_number(self, n):
        self.iter = n

    def set_clusters_number(self, n):
        self.K = n

    def __train(self, today):
        # efficient implementation using alpha/beta algorithm (Bishop, chap. 13.2.2)
        data = self.training_data.loc[:today,:].tail(self.window)
        epsilon = .00001
        L = data.shape[0]
        dim = data.shape[1]

        # inits
        p0 = (np.ones((self.K, 1)) / self.K).ravel()
        ah = np.ones((self.K, L)) / self.K  # fwd message
        bh = np.ones((self.K, L)) / self.K  # bkw message

        A = self.__init_A(self.p, self.K)  # state transition matrix
        mu = 0.1 * np.random.rand(dim, self.K)  # mu
        sa = np.zeros((dim, dim, self.K))  # covariance matrix
        for k in range(0, self.K):
            sa[:, :, k] = np.eye(dim)

        for j in range(0, self.iter):
            print('HMM iteration : {}'.format(j))

            # E STEP ##################################################################################################
            em = np.zeros((self.K, L))
            for k in range(0, self.K):
                for i in range(0, L):
                    # emission probabilities
                    em[k, i] = multivariate_normal.pdf(data.iloc[i, :].values, mean=mu[:, k], cov=sa[:, :, k])

            # forward message
            ah[:, 0] = p0 * em[:, 0] / np.matmul(p0.T, em[:, 0])
            c = np.ones((L, 1))
            for i in range(1, L):
                aux = em[:, i] * np.matmul(A.T, ah[:, i - 1])
                c[i] = np.sum(aux)
                if c[i] == 0:
                    print('scaling factor 0')
                ah[:, i] = aux / c[i]

            # backward
            bh[:, L - 1] = np.ones(self.K)
            for i in range(L - 2, -1, -1):
                if c[i + 1] == 0:
                    print('scaling factor 0')
                bh[:, i] = np.matmul(A, em[:, i + 1] * bh[:, i + 1] / c[i + 1])

            g = np.zeros((self.K, L))
            h = np.zeros((self.K, self.K, L))
            for i in range(0, L):
                g[:, i] = ah[:, i] * bh[:, i]
                if i > 0:
                    for q in range(0, self.K):
                        for p in range(0, self.K):
                            h[q, p, i] = ah[q, i - 1] * em[p, i] * A[q, p] * bh[p, i] / c[i]

            # M STEP ##################################################################################################
            p0 = g[:, 0] / np.sum(g[:, 0])
            se = sa

            for k in range(0, self.K):
                mu[:, k] = np.matmul(data.values.T, g[k, :].T) / np.max((np.sum(g[k, :]), epsilon))
                for i in range(0, L):
                    se[:, :, k] = se[:, :, k] + g[k, i] * np.matmul(
                        np.atleast_2d((data.iloc[i, :].values - mu[:, k])).T,
                        np.atleast_2d((data.iloc[i, :].values - mu[:, k])))
                sa[:, :, k] = se[:, :, k] / np.max((np.sum(g[k, :]), epsilon))

            for q in range(0, self.K):
                for p in range(0, self.K):
                    A[q, p] = h[q, p, :].sum() / np.max((h[q, :, :].sum(), epsilon))

            maxInd = np.argmax(g, axis=0)

        self.g = g
        self.states = maxInd
        self.A = A

    def load_file(self, filename):
        #IMPORTANT: the file must be a csv with the same format as the file "markets_new.csv"
        df = pd.read_csv(filename, delimiter=',')

        df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))

        # Adding indices to the data frame
        print('Adding indices to the data frame')

        for file in ['VIX', 'FVX', 'GSPC', 'GDAXI']:
            df1 = pd.read_csv(file + '.csv', delimiter=',')

            df1 = pd.DataFrame(data=df1.values, columns=df1.columns,
                               index=pd.to_datetime(df1['Date'], format='%Y-%m-%d'))
            df1 = pd.DataFrame(df1['Close']).rename(columns={"Close": file})
            df0 = df1.join(df0, on='Date')

        df0 = df0.reindex(self.dtindex)
        df0 = df0.drop(columns=['Date'])
        df0 = df0.pct_change().fillna(0).iloc[1:,:]

        self.training_data = df0[['VIX', 'FVX', 'GSPC', 'GDAXI']]
        self.tradable_assets = df0.drop(columns=['VIX', 'FVX', 'GSPC', 'GDAXI'])



    def predict(self, today):
        data = self.tradable_assets.loc[:today,:].tail(self.window)
        L = data.shape[0]

        dim_tradable = data.shape[1]
        mu_tradable = 0.1 * np.random.rand(dim_tradable, self.K)  # mu

        for k in range(self.K):
            dataLocal = data.loc[self.states == k, :]

            if dataLocal.shape[0] > 1:
                mu_tradable[:, k] = self.__by_mean(
                    dataLocal)
            else:
                mu_tradable[:, k] = 0

        pred = np.matmul(mu_tradable, np.matmul(self.A.T, self.g[:, L - 1]))

        return pred

    def optimize(self, today, x, ra):

        self.__train(today)
        mu_pred = self.predict(today)
        #posteriori_prob, mu_s, cov_s, pred, states = hmm.expectation_maximization_mod(self.training_data, self.tradable_assets, self.K, iter, p)
        cov = ra * pd.DataFrame(data=cv.oas(x)[0], index=x.columns, columns=x.columns).fillna(0)

        problem = osqp.OSQP()
        k = len(mu_pred)

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

        problem.setup(sCov, -mu_pred, sA, l, u, verbose=False)

        # Solve problem
        res = problem.solve()
        pr = pd.Series(data=res.x, index=x.columns)
        return pr

    def run(self, risk_aversion, window, rebalancing_period):
        if not weekmask:
            window = window * 5
            rebalancing_period = rebalancing_period * 5
        self.window = window


        rebalancing_dates = self.dtindex[window - 1::rebalancing_period]
        input_returns = self.tradable_assets
        weights = pd.DataFrame(data=np.nan, columns=input_returns.columns, index=input_returns.index)
        for date in self.dtindex[window - 1:]:
            today = date
            returns = input_returns.loc[:today, :].tail(window)
            last = returns.index[-2]

            if today in rebalancing_dates:  # re-optimize and get new weights
                print('{}: Rebalancing weights'.format(today))
                weights.loc[today, :] = self.optimize(today, returns, risk_aversion)

            else:  # no re-optimization, re-balance the weights
                print('{}: relocate assets to preserve wights'.format(today))
                weights.loc[today, :] = weights.loc[last, :] * (1 + returns.loc[today, :]) \
                                        / (1 + (weights.loc[last, :] * returns.loc[today, :]).sum())

        self.pnl = (weights.shift(1) * input_returns).sum(axis=1)

    def get_metrics(self):
        # Indicators calculation
        if self.weekmask:
            n = 52
        else:
            n = 252

        # Max drawdown
        md = self.pnl.cumsum()[self.window:]
        Roll_Max = md.rolling(window=md.shape[0], min_periods=1).max()
        Daily_Drawdown = md - Roll_Max
        mdd = -Daily_Drawdown.min()

        # Sharpe
        sharpe = self.pnl.mean() / self.pnl.std() * np.sqrt(n)

        # Return
        ret = self.pnl.cumsum().iloc[-1]

        # Annualized return
        pnl_shape = self.pnl.cumsum().shape[0]
        ann_return = (1 + self.pnl.cumsum().iloc[-1]) ** (n / pnl_shape) - 1

        return {"sharpe": sharpe, "return": ret, "mdd": mdd, "ann_return": ann_return}



    def __init_A(self, p, K):
        return np.ones((K, K)) * p + np.eye(K) * (1 - K*p)

    def __by_mean(self, X):
        Factor = .01
        sigma = np.max(X.std())
        sigma_0 = Factor * sigma
        m = X.shape[0]
        mu_0 = np.mean(X.mean() / X.std())
        return ((np.sum(X, axis=0) / sigma ** 2 + mu_0 / sigma_0 ** 2) / (
                m / sigma ** 2 + 1 / sigma_0 ** 2)).values


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("No stocks file specified. Aborting.")
        exit()
    else:
        filename = sys.argv[1]

    # SETTINGS
    risk_aversion = 1

    # Note: window and rebalancing_period always expressed in weeks
    window = 104
    rebalancing_period = 12

    # Note: start date is the the first day you want to take into account in the whole process
    # notice that the actual first trading day is start_date + window * 5
    start_date = '2012-12-31'
    end_date = '2015-12-28'
    weekmask = False

    pf_optimizer = HMMPortfolioOptimizer(start_date, end_date, weekmask)

    #Note: the file must be a csv with the same format as the file "markets_new.csv"
    pf_optimizer.load_file(filename)

    pf_optimizer.run(risk_aversion, window, rebalancing_period)

    results = pf_optimizer.get_metrics()

    print('\n============ RESULTS ============')
    print('\nSharpe Ratio: {:.3f}\nMax DD: {:.3f}\nTotal return: {:.3f}\nAnnualized return:{:.3f}'. \
          format(results['sharpe'], results['mdd'], results['return'], results['ann_return']))
