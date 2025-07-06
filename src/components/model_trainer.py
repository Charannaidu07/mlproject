import os
import sys
from dataclasses import dataclass
# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.model_selection import RandomizedSearchCV

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], 
                train_array[:,-1], 
                test_array[:,:-1], 
                test_array[:,-1]
            )

            models = {
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'SVR': SVR(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'XGBRegressor': XGBRegressor(eval_metric='rmse')
            }

            model_report:dict=evaluate_models(X_train=X_train,
                                            y_train=y_train,
                                            X_test=X_test,
                                            y_test=y_test,
                                            models=models
                                             )
            logging.info(f"Model Report: {model_report}")
            #the best model from the report
            best_model_score= max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model training completed successfully")
            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)