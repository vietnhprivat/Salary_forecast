import numpy as np
from matplotlib import pyplot as plt

# Initial salary
initial_salary = 45000
n = 10
trials = 100

# Define probabilities for changes
job_change = 0.2
promotion = 0.2
stay = 0.6
change_probabilities = (job_change, promotion, stay)

# Salary increase
job_change_growth = 1.20
promotion_growth = 1.10
stay_growth = 1.04
salary_growth = (job_change_growth, promotion_growth, stay_growth)

# Simulator
groups = [0,1,2]
samples = np.random.choice(groups, size=(trials,n), p=change_probabilities)
growth_factors = np.array(salary_growth)[samples]
growth_factors = np.insert(growth_factors, 0,1, axis=1)
salary_trajectory = np.cumprod(growth_factors, axis=1)
# print(salary_trajectory)
samples.shape

# plt.plot(salary_trajectory.T, alpha=0.7)
# plt.show()

mu = np.mean(salary_trajectory, axis=0)
sigma = np.std(salary_trajectory, axis=0)
x = range(n+1)
plt.plot(x, mu, label="Mean salary", color="blue")
plt.plot(x, mu+sigma, label="Mean std", color="red")
plt.plot(x, mu-sigma, color="red")
plt.grid()
plt.legend()
plt.show()