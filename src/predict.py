from parameters import OUTCOME_COLUMN_NAME
from model_io import load_latest_model, load_model


def predict_df_closure(model):

    def predict_df(data_df):
        preds = model.predict(data_df)
        return pd.DataFrame({OUTCOME_COLUMN_NAME: preds})  

    return predict_df      
    
def predict_df(data_df, model):
    preds = model.predict(data_df)
    return pd.DataFrame({OUTCOME_COLUMN_NAME: preds})

# TODO: totally depends on case and what model expects.
def predict_single_case(case, model):
    df_wrapped_case = pd.DataFrame({'CASE': [case]})
    preds = model.predict(df_wrapped_case)
    return preds[0]

def predict_single_case_closure(model):
    return lambda case: predict_single_case(case, model)

def predict_single_case_latest_model(case):    
    model = load_latest_model()
    return predict_single_case(case, model)

def build_latest_model_single_case_predict_fn():
    model = load_latest_model()
    return predict_single_case_closure(model)


