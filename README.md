Salary Forecast — Career trajectory and salary Monte Carlo simulation

Overview

This repository contains tools to simulate career progression and forecast salary trajectories using Monte Carlo simulation. The model simulates yearly career events (new job, promotion, stay) and uses event-based growth factors to build salary trajectories across many simulated career paths.

Contents

- src/
  - Advanced_salary_forecast.py — Core simulation code (CareerSimulator, MC_simulation, salary_prediction)
  - MC_salary_forecast.py — Helper plotting utilities used by the notebook (plot_mean_std)
  - transition_probabilities.py — Transition matrices used by the simulator
  - data_analysis.ipynb — Notebook with exploratory analysis and visualizations

- test/
  - test_advanced_salary_forecast.py — Unit tests for core simulation functionality

Quickstart

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the main packages used:

```bash
pip install numpy matplotlib pytest
```

2. Run unit tests:

```bash
pytest -q
```

3. Run the analysis notebook

Open the notebook `src/data_analysis.ipynb` with Jupyter Lab or Notebook and run the cells. The notebook imports functions from `src/Advanced_salary_forecast.py` and performs Monte Carlo simulations and plotting.

How it works

- `CareerSimulator` models a single career. It stores yearly `event_history` and `level_history` and supports tenure-based adjustments to transition probabilities.
- `MC_simulation` runs many `CareerSimulator` instances and returns arrays of event histories and seniority level histories.
- `salary_prediction` converts event histories to random growth factors (sampled from distributions per event type), applies seniority-related constraints (steady, minimal growth after reaching max seniority), and returns simulated salary trajectories.

Development notes

- The notebook `src/data_analysis.ipynb` acts as the primary exploration interface and imports the simulation functions rather than redefining them inline.
- Tests are kept in `test/` and use the project `src` package import style. Ensure the project root is in `PYTHONPATH` when running tests from an IDE.

Contributing

If you'd like to contribute:
- Fork the repo
- Create a feature branch
- Add tests for new behavior
- Open a pull request with a clear description of the change

License

This repository does not include a license file. Add one if you plan to share the code publicly.
