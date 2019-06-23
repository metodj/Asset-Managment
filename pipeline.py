import optimize_pf as opt
import numpy as np

methods = ['equal_weights', 'HMM', 'HMM_with_VIX', 'js_mean', 'by_mean']
risk_aversion = 10000
window = 10

results = np.zeros([5, 4])
for method in methods:
    results[methods.index(method),:] = opt.run_pipeline(method, risk_aversion, window)

for m in methods:
    print(m)
    print(results[methods.index(m),:])