import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

np.random.seed(42)
def f(x):
    return np.exp(-(x-75)**4 / 5000) + 0.01 * np.sin(0.3 * x)

def metropolis_hastings(target, n_samples, x0=90, proposal_std=10, bounds=(0, 100)):
  
    samples = np.zeros(n_samples)
    current = x0  
    accepted = 0
    
    for i in range(n_samples):
        proposed = np.random.normal(current, proposal_std)
        
        if proposed < bounds[0] or proposed > bounds[1]:
            samples[i] = current
            continue
            
        ratio = target(proposed) / target(current)
        
        if np.random.rand() < ratio:
            current = proposed
            accepted += 1
            
        samples[i] = current
    
    acceptance_rate = accepted / n_samples
    print(f"Acceptance rate: {acceptance_rate:.2f}")
    return samples

n_samples = 10000
samples_task1 = metropolis_hastings(f, n_samples, proposal_std=5)

mean_task1 = np.mean(samples_task1)
var_task1 = np.var(samples_task1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(samples_task1, bins=50, density=True, alpha=0.7)
x_vals = np.linspace(0, 100, 1000)
plt.plot(x_vals, f(x_vals)/quad(f, 0, 100)[0], 'r-', label='Normalized target')
plt.title('Sample histogram vs true distribution')
plt.xlabel('Exam score')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(samples_task1, alpha=0.5)
plt.title('Trace plot')
plt.xlabel('Iteration')
plt.ylabel('Exam score')

plt.tight_layout()
plt.show()

print(f"Sample mean: {mean_task1:.2f}")
print(f"Sample variance: {var_task1:.2f}")



print('''
......................................................
      ............................................
      ............................................
      ............................................
      ..............................................

    ''')

# def estimate_normalizing_constant(samples, target, bounds):
   
#     x_uniform = np.random.uniform(bounds[0], bounds[1], size=len(samples))
    
#     Z_estimate = (bounds[1] - bounds[0]) * np.mean(target(x_uniform))
    
#     return Z_estimate

# Z_estimate = estimate_normalizing_constant(samples_task1, f, (0, 100))

# Z_numerical, _ = quad(f, 0, 100)

# print(f"Estimated normalizing constant (Z): {Z_estimate:.4f}")
# print(f"Numerically integrated Z: {Z_numerical:.4f}")
# print(f"Relative error: {abs(Z_estimate - Z_numerical)/Z_numerical:.2%}")







# print('''
# .......................................................
#       ..............................................
#       ...............................................
#       .................


# ''')


# data = pd.read_csv('bayes_linear_data.csv')
# x = data['x'].values
# y = data['y'].values

# def log_prior(theta):
#     return -theta**4 + 2*theta

# def log_likelihood(theta, x, y):
#     return -0.5 * np.sum((y - theta * x)**2)

# def log_posterior(theta, x, y):
#     return log_prior(theta) + log_likelihood(theta, x, y)

# def metropolis_hastings_bayesian(log_target, n_samples, theta0=1.0, proposal_std=0.5, args=()):
 
#     samples = np.zeros(n_samples)
#     current = theta0
#     accepted = 0
    
#     for i in range(n_samples):
#         proposed = np.random.normal(current, proposal_std)
        
#         log_ratio = log_target(proposed, *args) - log_target(current, *args)
        
#         if np.log(np.random.rand()) < log_ratio:
#             current = proposed
#             accepted += 1
            
#         samples[i] = current
    
#     acceptance_rate = accepted / n_samples
#     print(f"Acceptance rate: {acceptance_rate:.2f}")
#     return samples

# n_samples = 10000
# samples_task3 = metropolis_hastings_bayesian(log_posterior, n_samples, args=(x, y))

# burn_in = 1000
# thinning = 5
# posterior_samples = samples_task3[burn_in::thinning]

# posterior_mean = np.mean(posterior_samples)

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.hist(posterior_samples, bins=50, density=True, alpha=0.7)
# plt.title('Posterior distribution of θ')
# plt.xlabel('θ')
# plt.ylabel('Density')

# plt.subplot(1, 2, 2)
# plt.plot(samples_task3, alpha=0.5)
# plt.title('Trace plot of θ')
# plt.xlabel('Iteration')
# plt.ylabel('θ')

# plt.tight_layout()
# plt.show()

# print(f"Posterior mean of θ: {posterior_mean:.4f}")

# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, label='Data')
# plt.plot(x, posterior_mean * x, 'r-', label=f'Posterior mean: θ = {posterior_mean:.2f}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Data with regression line using posterior mean of θ')
# plt.legend()