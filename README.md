Salary Forecast — Career trajectory and salary Monte Carlo simulation

Overview

This repository contains tools to simulate career progression and forecast salary trajectories using Monte Carlo simulation. The model simulates yearly career events such as new job, promotion, and stay, and it uses event-based growth factors to construct salary trajectories across many simulated career paths.

Contents

1. src/
  1. Advanced_salary_forecast.py — Core simulation code (CareerSimulator, MC_simulation, salary_prediction)
  2. MC_salary_forecast.py — Helper plotting utilities (plot_mean_std)
  3. transition_probabilities.py — Transition matrices used by the simulator

2. test/
  1. test_advanced_salary_forecast.py — Unit tests for core simulation functionality

Quickstart

1. Create and activate a virtual environment (recommended).

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

  If you do not have a `requirements.txt`, install the main packages used:

  ```bash
  pip install numpy matplotlib pytest
  ```

2. Run unit tests.

  ```bash
  pytest -q
  ```

How it works

1. `CareerSimulator` models a single career. It records yearly `event_history` and `level_history` and supports tenure-based adjustments to transition probabilities.
2. `MC_simulation` runs many `CareerSimulator` instances and returns arrays of event histories and seniority level histories.
3. `salary_prediction` converts event histories to sampled growth factors per event type, enforces a minimum growth, and caps post-max-seniority increases at 1% per year.

Development notes

1. The analysis notebook was removed; simulation and plotting functions remain in the `src` package and are intended to be imported directly in scripts.
2. Tests are located in `test/`. When running tests from an IDE, ensure the repository root is on `PYTHONPATH`.

Contributing

If you would like to contribute, fork the repository, create a feature branch, add tests for new behavior, and open a pull request with a clear description of the change.

License

This repository does not include a license file. Add one if you plan to share the code publicly.
