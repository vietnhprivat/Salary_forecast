import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define next step
def next_state(current_state, transition_matrix):
    return np.random.choice([0, 1, 2], p=transition_matrix[current_state])

def simulate(n, current_state, transition_matrix):
    forecast = [current_state]
    for _ in range(n):
        current_state = next_state(current_state, transition_matrix)
        forecast.append(int(current_state))
    return forecast

def plot_mean_std(x, mu, sigma):
    plt.plot(x, mu, color="blue")
    plt.plot(x, mu+sigma, color="red")
    plt.plot(x, mu-sigma, color="red")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Define states
    states = ["New_job", "Promotion", "Stay"]
    salary_growth = [1.15, 1.1, 1.04]

    # Define probabilities for next state in each state
    transition_matrix = np.array([
        [0.0, 0.0, 1.0], # state 0 new job 
        [0.15, 0.1, 0.75], # state 1 promotion
        [0.25, 0.25, 0.5] # state 2 stay
    ])

    # Simulate ten year salary projections N times
    N = 100
    years = 20
    n_simulations = np.asarray([simulate(years, 0, transition_matrix) for _ in range(N)])

    # Map to salary increase percentages
    initial_salary = 45_000
    growth_factors = np.array(salary_growth)[n_simulations]
    salary_trajectory = initial_salary * np.cumprod(growth_factors, axis=1)

    # Plotting
    mu = np.mean(salary_trajectory, axis=0)
    sigma = np.std(salary_trajectory, axis=0)
    x = range(years+1)
    plot_mean_std(x, mu, sigma)
    
    
    

