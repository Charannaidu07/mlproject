import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
#import dill
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    
    Raises:
    - CustomException: If there is an error during saving the object.
    """
    try:
        # dir_path= os.path.dirname(file_path)
        # os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    """
    Evaluate multiple regression models and return their performance metrics.
    
    Parameters:
    - X_train: Training feature set.
    - y_train: Training target variable.
    - X_test: Testing feature set.
    - y_test: Testing target variable.
    - models (dict): Dictionary of model names and their instances.
    
    Returns:
    - model_report (dict): Dictionary containing model names and their evaluation metrics.
    
    Raises:
    - CustomException: If there is an error during model evaluation.
    """
    try:
        report = {}
        for model_name, model in models.items():
            try:
                # Get parameters if available, otherwise use empty dict
                model_params = params.get(model_name, {})
                
                if model_params:  # Only do GridSearch if parameters are provided
                    gs = GridSearchCV(model, model_params, cv=3)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)
                
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_model_score
                
            except Exception as e:
                logging.warning(f"Error evaluating {model_name}: {str(e)}")
                report[model_name] = None
                
        return report
    except Exception as e:
        raise CustomException(e, sys)