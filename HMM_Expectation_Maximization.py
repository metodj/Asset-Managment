import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def init_A(p, K):
    # p: switching probability
    return np.ones((K, K)) * p + np.eye(K) * (1 - K*p)

def get_artificial_data(K, T, lmu, lcv):
    P = 50
    data = pd.DataFrame(0, index=range(0, T), columns=range(0, 2))
    for i in range(0, T):
        s = [j for j in range(0, K) if P*j/K <= np.mod(i,P) < P*(j+1)/K][0]
        data.iloc[i, :] = lmu[s] + np.matmul(np.linalg.cholesky(lcv[s]), np.random.randn(2)).T

    return data

def expectation_maximization(data, K, iter = 50):

    # efficient implementation using alpha/beta algorithm (Bishop, chap. 13.2.2)
    epsilon = .00001
    L = data.shape[0]
    dim = data.shape[1]

    # inits
    p0 = (np.ones((K, 1))/K).ravel()
    ah = np.ones((K, L))/K              # fwd message
    bh = np.ones((K, L))/K              # bkw message

    A = init_A(0.1, K)         # state transition matrix
    mu = 0.1*np.random.rand(dim, K)         # mu
    sa = np.zeros((dim, dim, K))        # covariance matrix
    for k in range(0, K):
        sa[:, :, k] = np.eye(dim)

    for j in range(0, iter):
        print(j)

        # E STEP ##################################################################################################
        em = np.zeros((K, L))
        for k in range(0, K):
            for i in range(0, L):
                # emission probabilities
                em[k, i] = multivariate_normal.pdf(data.iloc[i, :].values, mean=mu[:, k], cov=sa[:, :, k])

        # forward message
        ah[:, 0] = p0 * em[:, 0] / np.matmul(p0.T, em[:, 0])
        c = np.ones((L, 1))
        for i in range(1, L):
            aux = em[:, i] * np.matmul(A.T, ah[:, i-1])
            c[i] = np.sum(aux)
            if c[i] == 0:
                print('scaling factor 0')
            ah[:, i] = aux / c[i]

        # backward
        bh[:, L-1] = np.ones(K)
        for i in range(L-2, -1, -1):
            if c[i+1] == 0:
                print('scaling factor 0')
            bh[:, i] = np.matmul(A, em[:, i+1] * bh[:, i+1] / c[i+1])

        g = np.zeros((K, L))
        h = np.zeros((K, K, L))
        for i in range(0, L):
            g[:, i] = ah[:, i] * bh[:, i]
            if i > 0:
                for q in range(0, dim):
                    for p in range(0, dim):
                        h[q, p, i] = ah[q, i-1]*em[p, i]*A[q, p]*bh[p, i]/c[i]

        # M STEP ##################################################################################################
        p0 = g[:, 0] / np.sum(g[:, 0])
        se = sa

        for k in range(0, K):
            mu[:, k] = np.matmul(data.values.T, g[k, :].T) / np.max((np.sum(g[k, :]), epsilon))
            for i in range(0, L):
                se[:, :, k] = se[:, :, k] + g[k, i] * np.matmul(np.atleast_2d((data.iloc[i, :].values - mu[:, k])).T, np.atleast_2d((data.iloc[i, :].values - mu[:, k])))
            sa[:, :, k] =  se[:, :, k] / np.max((np.sum(g[k, :]), epsilon))

        for q in range(0, dim):
            for p in range(0, dim):
                A[q, p] = h[q, p, :].sum() / np.max((h[q, :, :].sum(), epsilon))

    return g, mu, sa


def multivariate_gaussian(pos, mu, Sigma):

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

if __name__ == '__main__':

    file = 'input_data.xlsx'
    data = pd.read_excel(file, 'input')
    dates = data['Date']
    ids = [1, 2, 6]
    data = data.iloc[:, ids]
    names = data.columns

    K = 3             # number of clusters
    iter = 50         # number of epochs (EM algo)

    posteriori_prob, mu_s, cov_s = expectation_maximization(data, K, iter=iter)

    # regimes
    plt.plot(posteriori_prob.T)

    # annotated underlyings
    maxInd = np.argmax(posteriori_prob, axis=0)
    sp = data.iloc[:, 0].cumsum()
    ty = data.iloc[:, 1].cumsum()
    vx = data.iloc[:, 2].cumsum()
    plt.figure()
    for i in range(0, K):
        nsp = np.nan * sp
        nty = np.nan * ty
        nvx = np.nan * vx
        nsp[maxInd == i] = sp[maxInd == i]
        nty[maxInd == i] = ty[maxInd == i]
        nvx[maxInd == i] = vx[maxInd == i]
        plt.subplot(311), plt.plot(nsp), plt.title('SPX')
        plt.subplot(312), plt.plot(nty), plt.title('T10')
        plt.subplot(313), plt.plot(nvx), plt.title('VIX')

    g = pd.DataFrame(posteriori_prob.T, index=dates, columns=range(0, K))
    g.to_csv('regimes.csv')
    plt.show()

    #     # unit test on artificial data
    #     lmu = (np.array([0.1, 0.07]), np.array([-0.15, 0.0]), np.array([0.00, 0.2]))
    #     vols = np.diag(np.array([0.1, 0.05]))
    #     lcv = (np.matmul(np.matmul(vols, np.array([(1, 0.9), (0.9, 1)])), vols), np.matmul(np.matmul(vols, np.array([(1, -0.9),(-0.9, 1)])), vols), np.matmul(np.matmul(vols, np.array([(1, 0.5), (0.5, 1)])), vols))
    #     data = get_artificial_data(3, 1500, lmu, lcv)
    #     iter = 10
    #     K = 3
    #     transition_prob = .03
    #     posteriori_prob, mu_s, cov_s = expectation_maximization(data, K, transition_prob, iter=iter)
    #     plt.plot(posteriori_prob.T)
    #     N = 60
    #     x = np.linspace(-.5, .5, N)
    #     y = np.linspace(-.5, .5, N)
    #     X, Y = np.meshgrid(x,y)
    #     pos = np.empty(X.shape + (2,))
    #     pos[:, :, 0] = X
    #     pos[:, :, 1] = Y
    #     fig = plt.figure()
    #     ax = fig.gca()
    #     Z = np.zeros((N,N))
    #     for i in range(0,K):
    #         Z = Z + multivariate_gaussian(pos, mu_s[:,i], cov_s[:,:,i])
    #     ax.contourf(X, Y, Z, offset=-0.15, cmap=cm.viridis)
    #     ax.scatter(data.iloc[:,0],data.iloc[:,1],color = 'r')
    #     plt.figure()
    #     plt.scatter(data.iloc[:,0],data.iloc[:,1])
    #     plt.show()
