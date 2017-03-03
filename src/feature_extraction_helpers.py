from sklearn.pipeline import FeatureUnion
class DfFeatureUnion(FeatureUnion):

    def __init__(self, df_transformers):
        self.df_transformers = df_transformers

    def transform(self, df):
        return pd.concat((transformer.transform(df) for transformer in self.df_transformers), axis = 1)
