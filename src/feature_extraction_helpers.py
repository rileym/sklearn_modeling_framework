import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion

class DfFeatureUnion(FeatureUnion):

    def __init__(self, df_transformers, n_jobs = -1):
        super(DfFeatureUnion, self).__init__(transformers = df_transformers, n_jobs = n_jobs)

    def _hstack_dfs(self, dfs):
        return pd.concat(dfs, axis = 1)

    def fit(self, X, y=None):
        """Fit all transformers using X.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
        Returns
        -------
        self : DfFeatureUnion
            This estimator
        """
        self._validate_transformers()
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, X, y)
            for _, trans, _ in self._iter())
        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
        Returns
        -------
        X_df : Dataframe result of union of transformed input.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, name, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        X_dfs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        return self._hstack_dfs(X_dfs)

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_df : Dataframe result of union of transformed input.
        """
        X_dfs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, name, weight, X)
            for name, trans, weight in self._iter())

        return self._hstack_dfs(X_dfs)
        
