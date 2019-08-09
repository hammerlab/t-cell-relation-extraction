import collections
import numpy as np
from tcre.env import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


def get_estimators(cv):

    def scorer(est, X, y_true):
        y_pred = np.squeeze(est.predict(X))
        y_true = np.squeeze(y_true)
        assert np.all(np.in1d(y_pred, [0, 1]))
        assert np.all(np.in1d(y_true, [0, 1]))
        if len(np.unique(y_pred)) < 2:
            return np.nan
        return f1_score(y_true, y_pred)

    def get_linear_model_gs(est):
        return GridSearchCV(est, param_grid=dict(C=np.logspace(-3, 2, 15)), cv=cv, scoring=scorer)

    ests = collections.OrderedDict([
        ('gbr', GridSearchCV(
            GradientBoostingClassifier(random_state=TCRE_SEED),
            param_grid=dict(
                n_estimators=[50, 100],
                learning_rate=[.1, .05, .01],
                max_depth=[1, 3, 5],
                min_samples_leaf=[1, 3]
            ),
            cv=cv,
            scoring=scorer
        )),
        ('xgb', GridSearchCV(
            XGBClassifier(random_state=TCRE_SEED),
            param_grid=dict(
                n_estimators=[50, 100],
                learning_rate=[.1, .05, .01],
                max_depth=[1, 3, 5]
            ),
            cv=cv,
            scoring=scorer
        )),
        ('ridge', Pipeline([
            ('normalize', StandardScaler()),
            ('est', get_linear_model_gs(LogisticRegression(random_state=TCRE_SEED, penalty='l2', solver='lbfgs')))
        ])),
        ('ridge2', Pipeline([
            ('normalize', StandardScaler()),
            ('feat', PolynomialFeatures(degree=2)),
            ('est', get_linear_model_gs(LogisticRegression(random_state=TCRE_SEED, penalty='l2', solver='lbfgs')))
        ])),
        ('lasso', Pipeline([
            ('normalize', StandardScaler()),
            ('est', get_linear_model_gs(LogisticRegression(random_state=TCRE_SEED, penalty='l1', solver='liblinear')))
        ])),
        ('lasso2', Pipeline([
            ('normalize', StandardScaler()),
            ('feat', PolynomialFeatures(degree=2)),
            ('est', get_linear_model_gs(LogisticRegression(random_state=TCRE_SEED, penalty='l1', solver='liblinear')))
        ])),

    ])
    return ests