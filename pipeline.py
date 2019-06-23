import optimize_pf_ale as opt
import numpy as np

methods = ['HMMLearn_expanded','HMMLearn', 'equal_weights', 'Sample_mean', 'HMM', 'HMM_expanded', 'js_mean', 'by_mean']
#methods = ['HMMLearn', 'Sample_mean', 'HMM', 'equal_weights']
risk_aversion = 1
window = 52

results = np.zeros([len(methods), 4])
for method in methods:
    results[methods.index(method),:] = opt.run_pipeline(method, risk_aversion, window)

for m in methods:
    print(m)
    print(results[methods.index(m),:])