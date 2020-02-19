# -*- coding: utf-8 -*-

import io
import pandas as pd
from numpy import nan, append, expand_dims, mean, std
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class NegativeRemoving(TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X, y=None):
      X[X<=0] = nan
      return X

    def fit(self, X, y=None):
        return self

class AddRatios(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X, y=None):
        X = append(X, expand_dims((X[:, 0] / X[:, 1]), 1), 1)
        X = append(X, expand_dims((X[:, 0] / X[:, 2]), 1), 1)
        X = append(X, expand_dims((X[:, 0] / X[:, 3]), 1), 1)
        X = append(X, expand_dims((X[:, 1] / X[:, 2]), 1), 1)
        X = append(X, expand_dims((X[:, 1] / X[:, 3]), 1), 1)
        X = append(X, expand_dims((X[:, 2] / X[:, 3]), 1), 1)
        return X

    def fit(self, X, y=None):
        return self

pd.options.mode.chained_assignment = None

df = pd.read_csv('Graduate - IRISES dataset (2019-06).csv', sep="|")
df['Petal.Width'] = df['Petal.Width'].str.replace(',', '.')
df['Petal.Width'] = pd.to_numeric(df['Petal.Width'], errors='raise')


y = df.Species
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
X = df.drop('Species', 1)

negative_removing = NegativeRemoving()
simple_imputer = SimpleImputer()
add_ratios = AddRatios()
model = XGBClassifier(n_estimators=500)

pipeline = Pipeline(steps=[
                           ("negative_removing", negative_removing),
                           ('simple_imputer', simple_imputer),
                           ('add_ratios', add_ratios),
                           ('model', model)
])

parameters = {
    "model__max_depth": [2, 3, 4, 5],
    "model__colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "model__subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
}
params_optimization = GridSearchCV(pipeline, parameters, 'accuracy', cv=5, iid=False)

scores = cross_val_score(params_optimization, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))