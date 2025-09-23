import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define next step
def next_state(current_state, transition_matrix, stay_streak, decay =[0.05, 0.05, -0.1]):
    # Stay streak
    if current_state == 2:
        base_probabilities = transition_matrix[2].copy()
        limit = transition_matrix[2][2]/-decay[2]
        adjustment = np.asarray(decay) * np.min([limit, stay_streak])
        adjusted_probabilities = base_probabilities[2] + adjustment
        
        # Sanity check: Sum of probabilites Alwyas result in 1.0 
        adjusted_probabilities = np.clip(adjusted_probabilities, 0, None)
        adjusted_probabilities /= adjusted_probabilities.sum()
    
        return np.random.choice([0, 1, 2], p=adjusted_probabilities)    
    
    else:
        return np.random.choice([0, 1, 2], p=transition_matrix[current_state])

# Define simulator
def simulate(n, current_state, transition_matrix, decay):
    forecast = [current_state]
    stay_streak = 0
    seniority = 1
    for _ in range(n):
        if current_state == 2:
            stay_streak += 1
        else:
            stay_streak = 0
            seniority += 1
            
        # Update state
        current_state = next_state(current_state, transition_matrix, stay_streak, decay)
        forecast.append(int(current_state))
    return forecast

# Plotting
def plot_mean_std(x, mu, sigma):
    plt.plot(x, mu, color="blue")
    plt.plot(x, mu+sigma, color="red")
    plt.plot(x, mu-sigma, color="red")
    plt.grid()
    plt.show()

# Define states and their probabilities
states = ["New_job", "Promotion", "Stay"]
salary_growth = [1.20, 1.1, 1.035]

# Define probabilities for next state in each state
transition_matrix = np.array([
    [0.025, 0.025, .95], # state 0 new job 
    [0.10, 0.05, 0.85], # state 1 promotion
    [0.10, 0.10, 0.8] # state 2 stay
])

# Simulate ten year salary projections N times
N = 1000
years = 10
decay = [0.05, 0.05, -0.1]
n_simulations = np.asarray([simulate(years, 0, transition_matrix, decay) for _ in range(N)])

# Map to salary increase percentages
initial_salary = 45_000
growth_factors = np.array(salary_growth)[n_simulations]
salary_trajectory = initial_salary * np.cumprod(growth_factors, axis=1)

# Plotting
mu = np.mean(salary_trajectory, axis=0)
sigma = np.std(salary_trajectory, axis=0)
x = range(years+1)
plot_mean_std(x, mu, sigma)

    
    

