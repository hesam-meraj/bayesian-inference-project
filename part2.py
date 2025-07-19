from part1 import samples,f,n_samples
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


# We start to use different methods 

'''
first method : IMPORTANT SAMPLING (IS) with uniform proposal
(using law of large numbers)


        proposal distribution
            q(x) = 1/100 x in [0,100]
    
    
    1 ) computing importance weight wi = f(xi)/q(xi) = 100.f(xi)
    2 ) Z_hat = 1/N sum(wi over 1 to N) = 100/N * sum(f(xi) over 1 to N)

'''
# n_samples = N
plt.hist(samples)
plt.show()
hist_counts, bin_edges = np.histogram(samples, bins=100, range=(0, 100), density=True)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Corresponding f(x) values
f_vals = f(bin_centers)

# Estimate Z as f(x) / estimated p(x) => average this over the domain
Z_estimates = f_vals / hist_counts
Z_est = np.mean(Z_estimates[np.isfinite(Z_estimates)])  # Avoid div-by-zero or inf
print(f"Estimated Z from samples: {Z_est:.4f}")


Z_numerical, _ = quad(f, 0, 100)
print(f"Numerically integrated Z: {Z_numerical:.4f}")
from scipy.stats import norm

