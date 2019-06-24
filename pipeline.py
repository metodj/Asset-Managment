import optimize_pf_ale as opt
import numpy as np
import pandas as pd
import HMM as hmm
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator


#methods = ['HMMLearn_expanded', 'HMM', 'HMM_mod', 'equal_weights', 'Sample_mean', 'HMM_expanded', 'js_mean', 'by_mean']
methods = ['HMM', 'HMM_mod', 'Sample_mean'] # HMM_expanded', 'Sample_mean', 'equal_weights']
risk_aversion = 1
window = 104
rebalancing_period = 12
dtindex = pd.bdate_range('2005-12-31', '2015-12-28', weekmask='Fri', freq='C')

results = np.zeros([len(methods), 5])



for method in methods:
    results[methods.index(method),:] = opt.run_pipeline(method, risk_aversion, window, rebalancing_period, dtindex)

for m in methods:
    print(m)
    print(results[methods.index(m),:])
"""

d1 = pd.bdate_range('1992-12-31', '1997-12-28', freq='C')
d2 = pd.bdate_range('1997-12-31', '2002-12-28', freq='C')
d3 = pd.bdate_range('2002-12-31', '2007-12-28', freq='C')
d4 = pd.bdate_range('2007-12-31', '2012-12-28', freq='C')

d = [d1, d2, d3, d4]
results = np.zeros([4, 5])

for range in d:
    for method in ["HMM_mod", "Sample_mean"]:
        print(opt.run_pipeline(method, risk_aversion, window, rebalancing_period, range))
"""