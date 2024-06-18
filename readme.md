# Automated Hyperparameter Optimization (HPO) System

## Introduction
The quality of performance of a Machine Learning model heavily depends on its hyperparameter settings. Given a dataset and a task, the choice of the machine learning (ML) model and its hyperparameters is typically performed manually.
The project is an automated hyperparameter optimization (HPO) system utilizing AutoML techniques to efficiently determine the optimal hyperparameter configuration for a given machine learning model and dataset.


## Project Structure

- `main.py`: The main script that runs the entire optimization and comparison process.
- `src/`: Directory containing the source code.
  - `BayesianOptimization.py`: Contains the implementation of the BayesianOptimizer class.
  - `Objective.py`: Defines the objective function for Bayesian Optimization.
  - `HyperoptObjective.py`: Defines the objective function for Hyperopt.
  - `CrossValidate.py`: Includes the cross-validation function.
  
## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Radhika811/AutoML_HPO.git
   cd AutoML_HPO
   ```

2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Optimization
Execute the main script to perform Bayesian and Hyperopt optimizations and compare their results:
```bash
python main.py
```

### Output
The system effectively integrates with various machine learning models and handles different data types. It employs efficient AutoML techniques like Bayesian optimization and random forests for hyperparameter optimization (HPO). The submission repository includes comprehensive results, featuring ROC AUC scores, cross-validation results, and comparisons of learning rate distribution curves with respect to Hyperopt (random vs submitted model vs Hyperopt scores) for thorough evaluation. 

## Code Overview

### Bayesian Optimization
The `BayesianOptimizer` class performs Bayesian Optimization. It requires:
- `func`: The objective function to be optimized.
- `float_param_ranges`: Dictionary specifying the ranges for continuous parameters.
- `int_param_candidates`: Dictionary specifying candidates for integer parameters.
- `n_init_points`: Number of initial points to sample.
- `max_iter`: Maximum number of iterations.
- `acq_type`: Acquisition function type ('EI' in this case).

### Cross-validation
The `cross_validate_with_params` function performs cross-validation using specified parameters and returns the ROC AUC scores.

### Plotting
The script generates plots using Matplotlib to visualize the distributions of ROC AUC scores and objective function values.

## Conclusion
This project demonstrates the use of Bayesian Optimization and Hyperopt for hyperparameter tuning of a RandomForestClassifier. By comparing the performance of these two techniques, the project provides insights into their effectiveness in finding optimal hyperparameters. The results include convergence plots, learning rate distributions, and cross-validation scores, offering a comprehensive analysis of the optimization process.
