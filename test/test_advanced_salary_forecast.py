import pytest
import numpy as np
from src.Advanced_salary_forecast import CareerSimulator

# Sample test data
@pytest.fixture
def sample_transition_matrix():
    # Simple 3x3x3 transition matrix for testing
    return {
        0: np.array([
            [0.3, 0.3, 0.4],
            [0.2, 0.3, 0.5],
            [0.2, 0.2, 0.6]
        ]),
        1: np.array([
            [0.3, 0.3, 0.4],
            [0.2, 0.3, 0.5],
            [0.2, 0.2, 0.6]
        ])
    }

@pytest.fixture
def sample_career(sample_transition_matrix):
    return CareerSimulator(
        initial_salary=45000,
        start_level=0,
        start_event=2,  # Stay
        transition_matrix=sample_transition_matrix,
        decay=[0.05, 0.05, -0.1]
    )

def test_career_simulator_initialization(sample_career):
    """Test if CareerSimulator initializes correctly"""
    assert sample_career.salary == 45000
    assert sample_career.seniority_level == 0
    assert sample_career.job_state == 2
    assert sample_career.year == 0
    assert sample_career.tenure == 0
    assert len(sample_career.salary_history) == 1
    assert len(sample_career.level_history) == 1
    assert len(sample_career.event_history) == 1

def test_tenure_adjustment(sample_career):
    """Test if tenure adjustment modifies probabilities correctly"""
    initial_probs = sample_career.transition_matrix[0][2]
    sample_career.tenure = 2
    adjusted_probs = sample_career.tenure_adjustment(sample_career.transition_matrix[0])
    
    # Check if probabilities still sum to 1
    assert np.isclose(sum(adjusted_probs), 1.0)
    # Check if probabilities were adjusted
    assert not np.array_equal(initial_probs, adjusted_probs)

def test_seniority_limit(sample_career):
    """Test if seniority limit works correctly"""
    # Test with current level
    assert sample_career.seniority_limit() <= max(sample_career.transition_matrix.keys())
    
    # Test with higher level
    sample_career.seniority_level = 5
    assert sample_career.seniority_limit() == max(sample_career.transition_matrix.keys())

def test_step_function(sample_career):
    """Test if step function updates state correctly"""
    initial_state = sample_career.job_state
    initial_level = sample_career.seniority_level
    
    sample_career.step()
    
    # Check that state has been updated
    assert sample_career.probabilities is not None
    assert isinstance(sample_career.job_state, (int, np.integer))
    assert 0 <= sample_career.job_state <= 2

def test_simulate_multiple_years(sample_career):
    """Test simulation over multiple years"""
    n_years = 5
    sample_career.simulate(n_years)
    
    # Check if histories are updated correctly
    assert len(sample_career.level_history) == n_years + 1  # +1 for initial state
    assert len(sample_career.event_history) == n_years + 1
    assert sample_career.year == n_years

    # Check if all events are valid
    assert all(0 <= event <= 2 for event in sample_career.event_history)
    
def test_tenure_counting(sample_career):
    """Test if tenure is counted and reset correctly"""
    # Force a "Stay" event
    sample_career.job_state = 2
    sample_career.step()
    
    if sample_career.job_state == 2:  # If stayed again
        assert sample_career.tenure > 0
    else:
        assert sample_career.tenure == 0

def test_probability_sanity(sample_career):
    """Test if probabilities always sum to 1"""
    for _ in range(5):  # Test for multiple steps
        sample_career.step()
        assert np.isclose(sum(sample_career.probabilities), 1.0)
        assert all(0 <= p <= 1 for p in sample_career.probabilities)