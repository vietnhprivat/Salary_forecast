import numpy as np
from matplotlib import pyplot as plt

class CareerSimulator:
    def __init__(
        self, initial_salary, start_level, start_event,
            transition_matrix, decay):
        
        # Constants
        self.salary = initial_salary
        self.seniority_level = start_level
        self.job_state = start_event
        self.year = 0
        self.tenure = 0
        self.probabilities = None

        # Rules
        self.transition_matrix = transition_matrix
        self.decay = decay
        
        # Initialize list for saving data for later
        self.salary_history = [self.salary]
        self.level_history = [self.seniority_level]
        self.event_history = [self.job_state]
        
    def tenure_adjustment(self, probability_matrix):
        """If staying at the same company the probability of getting promotion or new job increases """
        base_probabilities = probability_matrix[2].copy()
        limit = probability_matrix[2][2] / -self.decay[2] # Make sure you dont get negative values money
        factor = min(limit, self.tenure)
        adjusted_probabilities = base_probabilities + np.asarray(self.decay) * factor

        # Sanity check: Sum of probabilites always result in 1.0 
        adjusted_probabilities = np.clip(adjusted_probabilities, 0, None)
        adjusted_probabilities /= adjusted_probabilities.sum()
        
        return adjusted_probabilities
            
    def seniority_limit(self):
        """Makes sure that the seniority level doesn't go above the count of seniority probabilities defined in the transition matrix"""
        return min(max(self.transition_matrix), self.seniority_level)
    
    def next_step(self, probabilities):
        """Finds next step"""
        self.job_state = np.random.choice([0, 1, 2], p=probabilities)
        self.probabilities = probabilities
        
    
    def step(self):
        """Step function"""
        
        # Find senority level and match the following transition probabilities
        max_seniority = self.seniority_limit()
        probability_matrix = self.transition_matrix[max_seniority]
        
        # If on a stay streak (Been at the same company 2 years in a row)
        if (self.job_state == 2) and (self.seniority_level < max(self.transition_matrix)):
            # Adjust probabilities to account for longer tenure
            adjusted_probabilities = self.tenure_adjustment(probability_matrix)
            
            # Update state
            self.tenure += 1
            self.next_step(adjusted_probabilities)
            
        else:
            # Update state
            self.seniority_level += 1
            self.seniority_level = self.seniority_limit()
            self.tenure = 0
            probabilities = probability_matrix[self.job_state]
            self.next_step(probabilities)            
            
    def simulate(self, n_years):
        """Run the career simulation for n_years """
        for _ in range(n_years):
            self.step()
            self.year += 1

            # Save history
            self.level_history.append(self.seniority_level)
            self.event_history.append(int(self.job_state))

    def reset(self):
        """Reset to initial state"""
        # TODO Either delete or actually use it
        pass

## Simulation Functions
def MC_simulation(n_runs, n_years, **kwargs):
    all_event_histories = []
    all_level_histories = []

    for _ in range(n_runs):
        career = CareerSimulator(**kwargs)
        career.simulate(n_years)
        all_event_histories.append(career.event_history)
        all_level_histories.append(career.level_history)

    return np.array(all_event_histories), np.array(all_level_histories)

def salary_prediction(events, levels, initial_salary, growth_factor_distributions):
    """Calculate salary trajectories from event histories with max seniority adjustment
    
    Args:
        events: Array of event histories from MC simulation (n_runs x n_years+1)
        levels: Array of seniority levels from MC simulation
        initial_salary: Starting salary
        growth_factor_distributions: Dictionary of (mean, std) tuples for each event type
    """
    salary_growth_factors = np.zeros_like(events, dtype=float)
    n_runs, n_years_plus_one = events.shape
    
    # Find the first occurrence of max level for each career path
    max_level = np.max(levels)
    max_level_indices = np.argmax(levels == max_level, axis=1)
    
    for event_type, (mean, std) in growth_factor_distributions.items():
        # Select the corresponding event type
        event_positions = events == event_type
        
        # Draw from normal distribution
        n_occurrences = np.sum(event_positions)
        random_factors = np.random.normal(mean, std, size=n_occurrences)
        random_factors = np.maximum(random_factors, 1.01)
        
        # Assign the random factors to the corresponding positions
        salary_growth_factors[event_positions] = random_factors
    
    # TODO Start: Chat wrote this check it through
    # Override growth factors only AFTER the year max seniority is reached
    for i in range(n_runs):
        max_idx = max_level_indices[i]
        if max_idx < n_years_plus_one - 2:  # If max level was reached at least 2 years before the end
            # Keep the growth factor for the max seniority year, but fix all subsequent years to 1.01
            salary_growth_factors[i, max_idx+2:] = 1.01
    
    # Calculate cumulative salary trajectories
    salary_trajectories = initial_salary * np.cumprod(salary_growth_factors, axis=1)
    
    return salary_trajectories, salary_growth_factors

if __name__ == '__main__':
    from transition_probabilities import transition_matrices, transition_matrices2
        
    # Define growth factor distributions (mean, std) for each event type
    growth_factor_distributions = {
        0: (1.12, 0.05),  # New job: mean and std
        1: (1.08, 0.03),  # Promotion: 
        2: (1.03, 0.01),  # Stay: 
    }

    # Run 100 career simulations, each 15 years long or (rong as the japanese say)
    n_runs = 500
    n_years = 35
    initial_salary = 40_000
    transition_matrix = transition_matrices2
    decay = [0.025,0.025,-0.05]
    all_events, all_levels = MC_simulation(
        n_runs=n_runs,
        n_years=n_years,
        initial_salary=initial_salary,
        start_level=0,
        start_event=2,
        transition_matrix=transition_matrix,
        decay=decay
    )

    # Calculate salary trajectories with max seniority adjustment
    salary_trajectories, growth_factors = salary_prediction(
        events=all_events,
        levels=all_levels,
        initial_salary=initial_salary,
        growth_factor_distributions=growth_factor_distributions
    )

    # Plot 1: Salary trajectories
    plt.figure(figsize=(8, 4))
    x = np.arange(n_years + 1)
    mu = np.mean(salary_trajectories, axis=0)
    sigma = np.std(salary_trajectories, axis=0)
    
    plt.plot(x, mu, color="blue", label="Mean Salary")
    plt.fill_between(x, mu-sigma, mu+sigma, color="red", alpha=0.2, label="±1 std dev")
    plt.grid(True)
    plt.title("Salary Progression Over Time")
    plt.xlabel("Years")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()

    # Plot 2: Seniority levels
    plt.figure(figsize=(8, 4))
    mean = np.mean(all_levels, axis=0)
    std = np.std(all_levels, axis=0)
    
    plt.plot(x, mean, label="Mean Seniority Level")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std dev")
    plt.grid(True)
    plt.title("Seniority Level Progression")
    plt.xlabel("Years")
    plt.ylabel("Seniority Level")
    plt.legend()
    plt.show()
    