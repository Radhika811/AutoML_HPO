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
from Objective import objective_function
from HyperoptObjective import hyperopt_objective
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

#Bayesian Optimization
class Bayesian:
    def __init__(
            self,
            func,
            float_param_ranges={},
            int_param_candidates={},
            n_init_points=10000,
            external_init_points=None,
            max_iter=1e4,
            no_new_converge=3,
            no_better_converge=10,
            kernel=RBF(),
            acq_type='PI',
            beta_lcb=0.5,
            eps=1e-7,
            n_sample=int(1e6),
            seed=None
    ):
        self.func = func
        self.float_param_dict = float_param_ranges
        self.int_param_dict = int_param_candidates
        self.max_iter = int(max_iter)
        self.no_new_converge = no_new_converge
        self.no_better_converge = no_better_converge
        self.acq_type = acq_type
        self.beta_LCB = beta_lcb
        self.eps = eps
        self.n_sample = n_sample
        self.n_init_points = n_init_points
        self.seed = seed
        self.gpr = GPR(
            kernel=kernel,
            n_restarts_optimizer=50,
            random_state=self.seed
        )
        self.float_param_names = list(self.float_param_dict.keys())
        self.int_param_names = list(self.int_param_dict.keys())
        self.param_names = self.float_param_names + self.int_param_names
        self.float_param_ranges = np.array(list(self.float_param_dict.values()))
        self.int_param_candidates = list(self.int_param_dict.values())
        self.init_points = self.getInitPoints(external_init_points)
        self.x = self.init_points
        print('Evaluating Initial Points for Bayesian Optimization...')
        self.y = np.array(
            [self.func(**self.checkInitParams(dict(zip(self.param_names, p)))) for p in self.init_points]
        )
        u_index = self.uniqueIndex(self.x)
        self.x = self.x[u_index]
        self.y = self.y[u_index]
        self.num_param_seeds = len(self.x)
        self.gpr.fit(self.x, self.y)
        
    def getInitPoints(self, external_init_points):
        internal_init_points = self.generateRandomParams(self.n_init_points)
        if external_init_points is not None:
            nums = np.array([len(choices) for choices in external_init_points.values()])
            if not all(nums == nums[0]):
                raise Exception('Number of values for each parameter must be the same')
            if nums.sum() != 0:
                points = []
                for param in self.param_names:
                    points.append(external_init_points[param])
                points = np.array(points).T
                internal_init_points = np.vstack((internal_init_points, points))
        u_index = self.uniqueIndex(internal_init_points)
        return internal_init_points[u_index]

    def checkInitParams(self, param_dict):
        for k, v in param_dict.items():
            if k in self.int_param_names:
                param_dict[k] = int(param_dict[k])
        return param_dict

    def generateRandomParams(self, n):
        np.random.seed(self.seed)
        xs_range = np.random.uniform(
            low=self.float_param_ranges[:, 0],
            high=self.float_param_ranges[:, 1],
            size=(int(n), self.float_param_ranges.shape[0])
        )
        if len(self.int_param_dict) > 0:
            xs_candidates = np.array([np.random.choice(choice, size=int(n)) for choice in self.int_param_dict])
            xs_candidates = xs_candidates.T
            return np.hstack((xs_range, xs_candidates))
        else:
            return xs_range

    def uniqueIndex(self, xs):
        uniques = np.unique(xs, axis=0)
        if len(uniques) == len(xs):
            return list(range(len(xs)))
        counter = {tuple(u): 0 for u in uniques}
        indices = []
        for i, x in enumerate(xs):
            x_tuple = tuple(x)
            if counter[x_tuple] == 0:
                counter[x_tuple] += 1
                indices.append(i)
        return indices

    def aquisition(self, xs):
        print('Calculating utility Acquisition on sampled points based on GPR...')
        means, sds = self.gpr.predict(xs, return_std=True)
        sds[sds < 0] = 0
        z = (self.y.min() - means) / (sds + self.eps)
        if self.acq_type == 'EI':
            return (self.y.min() - means) * norm.cdf(z) + sds * norm.pdf(z)
        if self.acq_type == 'PI':
            return norm.pdf(z)
        if self.acq_type == 'LCB':
            return means - self.beta_LCB * sds

    def _min_acquisition(self, n=1e6):
        print('Random sampling based on ranges and candidates...')
        xs = self.generateRandomParams(n)
        ys = self.aquisition(xs)
        return xs[ys.argmin()]

    def optimize(self):
        no_new_converge_counter = 0
        no_better_converge_counter = 0
        best = self.y.min()
        for i in range(self.max_iter):
            print(f'Iteration: {i}, Current Best: {self.y.min()}')
            if no_new_converge_counter > self.no_new_converge:
                break
            if no_better_converge_counter > self.no_better_converge:
                break
            next_best_x = self._min_acquisition(self.n_sample)
            if np.any((self.x - next_best_x).sum(axis=1) == 0):
                no_new_converge_counter += 1
                continue
            print(f'Iteration {i}: evaluating guessed best param set by evaluation function...')
            self.x = np.vstack((self.x, next_best_x))
            next_best_y = self.func(**self.checkInitParams(dict(zip(self.param_names, next_best_x))))
            self.y = np.append(self.y, next_best_y)
            print(f'Iteration {i}: next best is {next_best_y}, {dict(zip(self.param_names, next_best_x))}')
            u_index = self.uniqueIndex(self.x)
            self.x = self.x[u_index]
            self.y = self.y[u_index]
            if self.y.min() < best:
                no_better_converge_counter = 0
                best = self.y.min()
            else:
                no_better_converge_counter += 1
            if len(self.x) == self.num_param_seeds:
                no_new_converge_counter += 1
            else:
                no_new_converge_counter = 0
                self.num_param_seeds = len(self.x)
            print(f'Iteration {i}: re-fit GPR with updated parameter sets')
            self.gpr.fit(self.x, self.y)

    def get_results(self):
        num_init = len(self.init_points)
        num_new = len(self.y) - num_init
        is_init = np.array([1] * num_init + [0] * num_new).reshape((-1, 1))
        results = pd.DataFrame(
            np.hstack((self.x, self.y.reshape((-1, 1)), is_init)),
            columns=self.param_names + ['AvgTestCost', 'isInit']
        )
        return results.sort_values(by='AvgTestCost', inplace=False)

#Cross validation
def cross_validate_with_hyperparameters(hyperparameters):
    num_estimators = int(hyperparameters['n_estimators'])
    tree_max_depth = int(hyperparameters['max_depth'])
    samples_split_min = int(hyperparameters['min_samples_split'])
    samples_leaf_min = int(hyperparameters['min_samples_leaf'])

    digit_data = load_digits()
    features, targets = digit_data.data, digit_data.target
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    rf_model = RandomForestClassifier(
        n_estimators=num_estimators,
        max_depth=tree_max_depth,
        min_samples_split=samples_split_min,
        min_samples_leaf=samples_leaf_min,
        random_state=42
    )

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(rf_model, scaled_features, targets, cv=k_fold, scoring='roc_auc_ovr')

    return auc_scores


float_param_ranges = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

int_param_candidates = {}

optimizer = Bayesian(
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
bayesian_cv_scores = cross_validate_with_hyperparameters(best_params_bayesian)

best_params_hyperopt = {
    'n_estimators': best['n_estimators'],
    'max_depth': best['max_depth'],
    'min_samples_split': best['min_samples_split'],
    'min_samples_leaf': best['min_samples_leaf']
}
hyperopt_cv_scores = cross_validate_with_hyperparameters(best_params_hyperopt)

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