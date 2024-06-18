import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from src.BayesianOptimization import BayesianOptimizer
from src.Objective import objective_function
from src.HyperoptObjective import hyperopt_objective
from src.CrossValidate import cross_validate_with_params
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

float_param_ranges = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

int_param_candidates = {}

optimizer = BayesianOptimizer(
    func=objective_function,
    float_param_ranges=float_param_ranges,
    int_param_candidates=int_param_candidates,
    n_init_points=100,
    max_iter=100,
    acq_type='EI'
)

optimizer.optimize()

results = optimizer.get_results()
print(results)

best_params = results.iloc[0]
print("Best Parameters:")
print(best_params)

# Convergence Plot
plt.plot(results['AvgTestCost'], color='green')
plt.xlabel('Iteration')
plt.ylabel('Negative Mean ROC AUC')
plt.title('Convergence Plot')
plt.show()

print("------------------------------------------------------------------")
     
space = {
    'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
    'max_depth': hp.quniform('max_depth', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1)
}

trials = Trials()
best = fmin(
    fn=hyperopt_objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("Best Parameters found by Hyperopt:")
print(best)

print("------------------------------------------------------------------")

bayesian_roc_auc_scores = results['AvgTestCost'].apply(lambda x: -x).values
hyperopt_roc_auc_scores = [-trial['result']['loss'] for trial in trials.trials]

# Bayesian versus Hyperopt Learning Rate Distribution
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.hist(bayesian_roc_auc_scores, bins=30, alpha=0.7, label='Bayesian Optimizer', color='green')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.title('Bayesian Optimizer Learning Rate Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(hyperopt_roc_auc_scores, bins=30, alpha=0.7, label='Hyperopt')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.title('Hyperopt Learning Rate Distribution')
plt.legend()

plt.tight_layout()
plt.show()

print("Best ROC AUC Score from Bayesian Optimizer: ", max(bayesian_roc_auc_scores))
print("Best ROC AUC Score from Hyperopt: ", max(hyperopt_roc_auc_scores))

print("------------------------------------------------------------------")

best_params_bayesian = results.iloc[0][['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']].to_dict()
bayesian_cv_scores = cross_validate_with_params(best_params_bayesian)

best_params_hyperopt = {
    'n_estimators': best['n_estimators'],
    'max_depth': best['max_depth'],
    'min_samples_split': best['min_samples_split'],
    'min_samples_leaf': best['min_samples_leaf']
}
hyperopt_cv_scores = cross_validate_with_params(best_params_hyperopt)

print("Cross-validation ROC AUC scores for Bayesian Optimizer best parameters: ", bayesian_cv_scores)
print("Cross-validation ROC AUC scores for Hyperopt best parameters: ", hyperopt_cv_scores)
print("Mean ROC AUC for Bayesian Optimizer: ", np.mean(bayesian_cv_scores))
print("Mean ROC AUC for Hyperopt: ", np.mean(hyperopt_cv_scores))

data = load_digits()
X, y = data.data, data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# RandomForestClassifier
default_model = RandomForestClassifier(random_state=42)

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
default_scores = cross_val_score(default_model, X, y, cv=cv, scoring='roc_auc_ovr')

default_mean_auc = np.mean(default_scores)

print("Cross-validation ROC AUC scores for the default RandomForestClassifier:", default_scores)
print("Mean ROC AUC for the default RandomForestClassifier:", default_mean_auc)

bayesian_roc_auc_scores = bayesian_cv_scores
hyperopt_roc_auc_scores = hyperopt_cv_scores

plt.figure(figsize=(14, 7))

# Bayesian Optimizer results
plt.subplot(1, 3, 1)
plt.hist(bayesian_roc_auc_scores, bins=10, alpha=0.7, label='Bayesian Optimizer', color='green')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.title('Bayesian Optimizer Learning Rate Distribution')
plt.legend()

# Hyperopt results
plt.subplot(1, 3, 2)
plt.hist(hyperopt_roc_auc_scores, bins=10, alpha=0.7, label='Hyperopt')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.title('Hyperopt Learning Rate Distribution')
plt.legend()

# Default Model results
plt.subplot(1, 3, 3)
plt.hist(default_scores, bins=10, alpha=0.7, label='Default Model', color='black')
plt.xlabel('ROC AUC Score')
plt.ylabel('Frequency')
plt.title('Default Model Learning Rate Distribution')
plt.legend()

#Bayesian versus Hyperopt versus Default model learning rate distribution
plt.tight_layout()
plt.show()

print("------------------------------------------------------------------")

bayesian_obj_values = results['AvgTestCost'].values
hyperopt_obj_values = [-trial['result']['loss'] for trial in trials.trials]

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.hist(bayesian_obj_values, bins=30, alpha=0.7, label='Bayesian Optimizer', color='green')
plt.xlabel('Objective Function Value')
plt.ylabel('Frequency')
plt.title('Bayesian Optimizer Objective Function Value Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(hyperopt_obj_values, bins=30, alpha=0.7, label='Hyperopt')
plt.xlabel('Objective Function Value')
plt.ylabel('Frequency')
plt.title('Hyperopt Objective Function Value Distribution')
plt.legend()

# Bayesian optimizer versus Hyperopt Objective Function Value Distribution 
plt.tight_layout()
plt.show()