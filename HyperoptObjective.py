from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def hyperopt_objective(hyperparameters):
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
    roc_auc_scores = cross_val_score(rf_model, scaled_features, targets, cv=k_fold, scoring='roc_auc_ovr')

    return {'loss': -np.mean(roc_auc_scores), 'status': STATUS_OK}
