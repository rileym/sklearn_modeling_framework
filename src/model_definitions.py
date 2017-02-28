from collections import namedtuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import numpy as np


ModelPackage = namedtuple('ModelPackage', 'name model param_grid')

# TODO: make recursive
def combine_pipeline_grids(step_name_grid_map):
    
    new_grid = dict()
    for step_name, grid in step_name_grid_map.iteritems():
        for param_name, params in grid.iteritems():
            new_grid[step_name + '__' + param_name] = params
            
    return new_grid

def make_tfidf_mode_pipeline(tfidf_vectorizer, tfidf_grid, estimator, estimator_grid):
    
    steps = [
        (TEXT_SELECTOR_STEP_NAME, ColumnToIterableTransformer(TEXT_COLUMN_NAME)),
        (TFIDF_STEP_NAME, tfidf_vectorizer),
        (MODEL_STEP_NAME, estimator)
    ]
    
    combined_grid = combine_pipeline_grids({TFIDF_STEP_NAME: tfidf_grid, MODEL_STEP_NAME:estimator_grid})
    return (Pipeline(steps = steps), combined_grid)

class ColumnToIterableTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X.loc[:, self.column].values                

# Example Set Up

tfidf_grid = dict(

            stop_words = [None, 'english'],
            ngram_range = [(1,1), (1,2), (2,2)],  
            min_df = [1,2]  ,
            binary = [True], 
            norm = [u'l1', None],
            use_idf = [True, False], 

)

tfidf_defaults = dict()

tfidf_vectorizer = TfidfVectorizer(**tfidf_defaults)


#
## Example Classifier Model Class Search Space
#

# Logistic Regression
logr_grid = dict(
        C = np.logspace(-5,5,50),
        class_weight = [None, 'balanced'],
        penalty = ['l1', 'l2']
    )
    
logr_defaults = dict(
                max_iter=1000
    )

logr_estimator = LogisticRegression(**logr_defaults)
logr_pipeline, logr_pipeline_param_grid = make_tfidf_mode_pipeline(tfidf_vectorizer, tfidf_grid, logr_estimator, logr_grid)
logr_model_package = ModelPackage(name = 'logistic_regression', model = logr_pipeline, param_grid = logr_pipeline_param_grid)


# Random Forest
rf_grid = dict(
            n_estimators=[2500],
            max_depth=[4, 8, None],
            min_samples_leaf=[3, 2, 1],
            max_features=['log2', 'sqrt'],
            class_weight=[None, 'balanced']
    )

rf_deaults = dict(
                bootstrap=True,
                n_jobs=-1,
                random_state=1492,
                verbose=2,
            )

rf_estimator = RandomForestClassifier(**rf_deaults)
rf_pipeline, rf_pipeline_param_grid = make_tfidf_mode_pipeline(tfidf_vectorizer, tfidf_grid, rf_estimator, rf_grid)
rf_model_package = ModelPackage(name = 'random_forest', model = rf_pipeline, param_grid = rf_pipeline_param_grid)


# SVM
svm_grid = dict(
            C=np.logspace(-3,3,50),
            kernel=['rbf'],
            gamma=np.logspace(-3,1,50),
    )

svm_deaults = dict(
                shrinking=True,
                random_state=1492,
                verbose=2,
    )

svm_estimator = SVC(**svm_deaults)
svm_pipeline, svm_pipeline_param_grid = make_tfidf_mode_pipeline(tfidf_vectorizer, tfidf_grid, svm_estimator, svm_grid)
svm_model_package = ModelPackage(name = 'svm', model = svm_pipeline, param_grid = svm_pipeline_param_grid)


# collect
model_packages = [logr_model_package, rf_model_package, svm_model_package]



