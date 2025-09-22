import numpy as np
from matplotlib import pyplot as plt

# Initial salary
initial_salary = 45000

# Define probabilities for changes
job_change = 0.4
promotion = 0.4
stay = 0.2
change_probabilities = (job_change, promotion, stay)

# Salary increase
job_change_growth = 1.20
promotion_growth = 1.10
stay_growth = 1.04
salary_growth = (job_change_growth, promotion_growth, stay_growth)

# Simulator
groups = [0,1,2]
samples = np.random.choice(groups, size=(100,10), p=change_probabilities)
growth_factors = np.array(salary_growth)[samples]
growth_factors = np.insert(growth_factors, 0,1, axis=1)
salary_trajectory = np.cumprod(growth_factors, axis=1)
# print(salary_trajectory)
samples.shape
# domain = range(len(salary_trajectory))
# codomain = salary_trajectory
plt.plot(salary_trajectory.T, alpha=0.7)