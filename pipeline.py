import optimize_pf_ale as opt
import numpy as np
import pandas as pd
import HMM as hmm
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator


methods = ['HMM_mod','HMM', 'Sample_mean'] # HMM_expanded', 'Sample_mean', 'equal_weights']
risk_aversion = 1
window = 252
rebalancing_period = 50
dtindex = pd.bdate_range('2005-12-31', '2015-12-28', freq='C')

results = np.zeros([len(methods), 5])



for method in methods:
    results[methods.index(method),:] = opt.run_pipeline(method, risk_aversion, window, rebalancing_period, dtindex)

for m in methods:
    print(m)
    print(results[methods.index(m),:])
