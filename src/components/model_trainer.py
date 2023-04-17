import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Model training initiated.')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info('input data splitted into training & testing data.')

            models = {
                'Linear Regression': LinearRegression(),
                'KNN Regressor': KNeighborsRegressor(),
                'DecisionTree Regressor': DecisionTreeRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'GradientBoost Regressor': GradientBoostingRegressor(),
                'RandomForest Regressor': RandomForestRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False),
                'XGBoost Regressor': XGBRegressor()
            }

            params={
                "DecisionTree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },

                "RandomForest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "GradientBoost Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "KNN Regressor":{},

                "Linear Regression":{},

                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_model(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                models = models,
                params = params
            )

            # To get best model score from model report
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f'Best model found \n model name: {best_model_name} \n score:{best_model_score}')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info("Model training object saved")

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            logging.info("Model training completed successfully.")

            return (model_report, best_model_name, r2)
        
        except Exception as e:
            raise CustomException(e, sys)