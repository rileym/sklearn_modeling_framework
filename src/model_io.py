import os
import datetime
import dill as pkl
from parameters import MODEL_BASE_DIR


# TODO: write tests

def get_current_date_as_str(fmt):
    date = datetime.datetime.now()
    return date.strftime(fmt)

def get_model_filename(model_name):
    
    date_fmt = '%Y-%m-%d-%H-%M'
    time_str = get_current_date_as_str(date_fmt)    
    filename_template = '{time}_{model_name}.{ext}'
    return filename_template.format(time = time_str, model_name = model_name, ext = 'pkl')

def get_model_path(base_dir, model_name):
    
    filename = get_model_filename(model_name)
    return os.path.normpath(os.path.join(base_dir, filename))
    
def get_all_model_filenames(model_dir, model_name = None):

    filenames = os.listdir(model_dir)
    model_filenames = filter( lambda filename: filename != '.gitinclude', filenames )
    filename_pred = (lambda filename: model_name in filename) if model_name else None
    return filter(filename_pred, model_filenames) 

def build_evaluation_output_path(dir_, model_name):
    return build_dated_named_path(dir_ = dir_, name = model_name, ext = 'txt')

def build_model_path(dir_, model_name):
    return build_dated_named_path(dir_ = dir_, name = model_name, ext = 'pkl')


def load_latest_model_(model_dir, name = None):
    
    model_filenames = get_all_model_filenames(model_dir, name)

    if not model_filenames:
        raise ValueError('No models found in {dir_}. Train a model first or move a model to {dir_}.'.format(dir_ = model_dir))

    latest_model_filename = sorted(model_filenames, reverse = True)[0]
    return load_model_(model_dir, latest_model_filename)

def save_model_(base_dir, model_name, model):
    path = get_model_path(base_dir, model_name)
    with open(path, 'wb') as f:
        pkl.dump(model, f)

def load_model_(model_dir, model_filename):

    load_path = os.path.normpath(os.path.join(model_dir, model_filename))    
    with open(load_path, 'rb') as f:
        return pkl.load(f)


# to be used externally

def load_model(model_filename):
    return load_model_(MODEL_BASE_DIR, model_filename) 

def load_latest_model(model_name = None):
    return load_latest_model_(MODEL_BASE_DIR, name = model_name)

def save_model(model_name, model):
    save_model_(MODEL_BASE_DIR, model_name, model)
    
