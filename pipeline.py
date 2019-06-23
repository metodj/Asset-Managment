import optimize_pf_ale as opt
import numpy as np

methods = ['equal_weights', 'Sample_mean', 'HMM', 'HMM_with_VIX', 'js_mean', 'by_mean']
#methods = ['HMM', 'HMM_with_VIX']
risk_aversion = 1
window = 120

results = np.zeros([len(methods), 4])
for method in methods:
    results[methods.index(method),:] = opt.run_pipeline(method, risk_aversion, window)

for m in methods:
    print(m)
    print(results[methods.index(m),:])