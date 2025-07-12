import numpy as np
import matplotlib.pyplot as plt
# importing this for computing the normalized constant
from scipy.integrate import quad

# non-negative function that we want to sample from 
def f(x):
    return np.exp(-(x-75)**4 / 5000) + 0.01 * np.sin(0.3 * x)


# metropolis algorithm

def metropolis_hastings(f, proposal_std, initial_x, n_samples):
    samples= [initial_x]
    accepted = 0
    
    
    # my proposal_distribution is normal to cancel out the accecpting prob
    # since normal is symmetric q(b|a) = q(a|b)
    # min(1, r_f * r_q) = min(1, r_f) because r_q = 1
    for i in range(n_samples):
        generated_number = np.random.normal(initial_x, proposal_std)
        
        if generated_number < 0 or generated_number > 100 :
            continue
        else:             
        # how much are they close in terms of target_dist
        # r_target_dist = target_dist(b) / target_dist(a)
        # here r stands for ratio 
            ratio = f(generated_number) / f(initial_x)
        
        
        # min(1, f(b)/f(a))
        # since normal is symmetric the prob of accepting will cancel out 
        
        accepting_prob = min(1, ratio)
        
        if accepting_prob == 1:
            initial_x = generated_number
            samples.append(generated_number)
            accepted += 1
        else:
            a = np.random.rand()
            if a <= accepting_prob:
                initial_x = generated_number
                samples.append(generated_number)
                accepted += 1
            else:
                continue
    acceptance_rate = accepted / n_samples
    print(f"Acceptance rate: {acceptance_rate:.2f}")

    return np.array(samples)

n_samples = 100000
proposal_std = 5
initial_x = 50

samples = metropolis_hastings(f, proposal_std, initial_x, n_samples)

mean_task1 = np.mean(samples)
var_task1 = np.var(samples)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.7)
x_vals = np.linspace(0, 100, 1000)
plt.plot(x_vals, f(x_vals)/quad(f, 0, 100)[0], 'r-', label='Normalized target')
plt.title('Sample histogram vs true distribution')
plt.xlabel('Exam score')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(samples, alpha=0.5)
plt.title('Trace plot')
plt.xlabel('Iteration')
plt.ylabel('Exam score')

plt.tight_layout()
plt.show()

print(f"Sample mean: {mean_task1:.2f}")
print(f"Sample variance: {var_task1:.2f}")
