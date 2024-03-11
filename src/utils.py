import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
def plus(self):
    a = 1+2
    return a
def minus(x):
    x-1
    return x
