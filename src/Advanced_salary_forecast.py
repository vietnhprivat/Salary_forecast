import numpy as np
from matplotlib import pyplot as plt
# Tue RF-learning kind of shit: 
class CareerSimulator:
    def __init__(
        self, initial_salary, start_level, start_event,
            transition_matrix, growth_factors, decay):
        
        # Constants
        self.salary = initial_salary
        self.seniority_level = start_level
        self.job_state = start_event
        self.year = 0
        self.tenure = 0
        self.probabilities = None

        # Rules
        self.transition_matrix = transition_matrix
        self.growth_factors = growth_factors 
        self.decay = decay
        
        # Data saver (stuff not working yet just place holders)
        self.salary_history = [self.salary]
        self.level_history = [self.seniority_level]
        self.event_history = [self.job_state]
        
    def tenure_adjustment(self, probability_matrix):
        """If staying at the same company the probability of getting promotion or leaving increases """
        base_probabilities = probability_matrix[2].copy()
        limit = probability_matrix[2][2] / -self.decay[2]
        factor = min(limit, self.tenure)
        adjusted_probabilities = base_probabilities + np.asarray(self.decay) * factor

        # Sanity check: Sum of probabilites always result in 1.0 
        adjusted_probabilities = np.clip(adjusted_probabilities, 0, None)
        adjusted_probabilities /= adjusted_probabilities.sum()
        
        return adjusted_probabilities
            
    def seniority_limit(self):
        """Trust me bro"""
        return min(max(self.transition_matrix), self.seniority_level)
    
    def step(self):
        """Step function help me (I am stuck)"""
        
        # Find senority level and match the following transition probabilities
        max_seniority = self.seniority_limit()
        probability_matrix = self.transition_matrix[max_seniority]
        
        # If on a stay streak (Been at the same company 2 years in a row)
        if (self.job_state == 2) and (self.seniority_level < max(self.transition_matrix)):
            # Adjust probabilities to account for longer tenure
            adjusted_probabilities = self.tenure_adjustment(probability_matrix)
            
            # Update state
            self.tenure += 1
            self.job_state = np.random.choice([0, 1, 2], p=adjusted_probabilities)
            self.probabilities = adjusted_probabilities
            
        else:
            # Update state
            self.seniority_level += 1
            self.seniority_level = self.seniority_limit()
            self.tenure = 0  
            self.job_state = np.random.choice([0, 1, 2], p=probability_matrix[self.job_state])
            
        self.probabilities = probability_matrix[self.job_state]

    def simulate(self, n_years):
        """Run the career simulation for n_years (If Mette-F is still in power n→∞)"""
        for _ in range(n_years):
            # Advance one year
            self.step()
            self.year += 1

            # Save history
            self.level_history.append(self.seniority_level)
            self.event_history.append(int(self.job_state))

    def reset(self):
        """Reset to initial state: Dunno Tue always includes this"""
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

if __name__ == '__main__':
    from src.transition_probabilities import transition_matrices
    
    # Salary increase percentages
    growth_factors = {
        "New_job": 1.20,
        "Promotion": 1.10,
        "Stay": 1.035,
    }

    # Initialize a carrer object
    career = CareerSimulator(
        initial_salary=45_000,
        start_level=0,
        start_event=2,  # 0=New_job, 1=Promotion, 2=Stay
        transition_matrix=transition_matrices,
        growth_factors=growth_factors,
        decay=[0.05, 0.05, -0.1]
    )

    # Run 100 career simulations, each 15 years long
    all_events, all_levels = MC_simulation(
        n_runs=100,
        n_years=15,
        initial_salary=45_000,
        start_level=0,
        start_event=2,
        transition_matrix=transition_matrices,
        growth_factors=growth_factors,
        decay=[0.05, 0.05, -0.1]
    )
    # Plot results
    print("Events shape:", all_events.shape)   
    print("Levels shape:", all_levels.shape)   
