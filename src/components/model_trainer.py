import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
import optuna
from sklearn.model_selection import cross_val_score
@dataclass
class ModelTrainerConfig:
    train_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_train(self, X_train, X_test, y_train, y_test):
        """
        Choose models and perform hyperparameter with optuna
        """
        def objective(trial):
            model_name = trial.suggest_categorical("model_name", ["Random Forest", "Decision Tree", "Gradient Boosting", "Linear Regression", "XGBRegressor", "CatBoosting Regressor", "AdaBoost Regressor"])

            if model_name == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int("rf_n_estimators", 8, 256),
                    criterion=trial.suggest_categorical("rf_criterion", ['mse', 'mae']),
                    max_features=trial.suggest_categorical("rf_max_features", ['auto', 'sqrt', 'log2', None])
                )
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor(
                    criterion=trial.suggest_categorical("dt_criterion", ['mse', 'friedman_mse', 'mae', 'poisson']),
                    splitter=trial.suggest_categorical("dt_splitter", ['best', 'random'])
                )
            elif model_name == "Gradient Boosting":
                model = GradientBoostingRegressor(
                    learning_rate=trial.suggest_float("gb_learning_rate", 0.001, 0.1),
                    n_estimators=trial.suggest_int("gb_n_estimators", 8, 256),
                    subsample=trial.suggest_float("gb_subsample", 0.6, 0.9),
                    criterion=trial.suggest_categorical("gb_criterion", ['friedman_mse', 'mse']),
                    max_features=trial.suggest_categorical("gb_max_features", ['auto', 'sqrt', 'log2'])
                )
            elif model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "XGBRegressor":
                model = XGBRegressor(
                    learning_rate=trial.suggest_float("xgb_learning_rate", 0.001, 0.1),
                    n_estimators=trial.suggest_int("xgb_n_estimators", 8, 256)
                )
            elif model_name == "CatBoosting Regressor":
                model = CatBoostRegressor(
                    depth=trial.suggest_int("cb_depth", 6, 10),
                    learning_rate=trial.suggest_float("cb_learning_rate", 0.01, 0.1),
                    iterations=trial.suggest_int("cb_iterations", 30, 100),
                    verbose=False
                )
            elif model_name == "AdaBoost Regressor":
                model = AdaBoostRegressor(
                    learning_rate=trial.suggest_float("ab_learning_rate", 0.001, 0.1),
                    n_estimators=trial.suggest_int("ab_n_estimators", 8, 256)
                )

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, cv=5).mean()

            return score

        # Set up the Optuna study and optimize the objective function
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        # Print the best parameters and score
        print("Best trial:")
        print("Value: ", study.best_trial.value)
        print("Params: ")
        for key, value in study.best_trial.params.items():
            print(f"{key}: {value}")

        # Instantiate the best model based on the Optuna results
        best_model_name = study.best_trial.params["model_name"]
        if best_model_name == "Random Forest":
            best_model = RandomForestRegressor(**study.best_trial.params)
        elif best_model_name == "Decision Tree":
            best_model = DecisionTreeRegressor(**study.best_trial.params)
        elif best_model_name == "Gradient Boosting":
            best_model = GradientBoostingRegressor(**study.best_trial.params)
        elif best_model_name == "Linear Regression":
            best_model = LinearRegression()
        elif best_model_name == "XGBRegressor":
            best_model = XGBRegressor(**study.best_trial.params)
        elif best_model_name == "CatBoosting Regressor":
            best_model = CatBoostRegressor(**study.best_trial.params, verbose=False)
        elif best_model_name == "AdaBoost Regressor":
            best_model = AdaBoostRegressor(**study.best_trial.params)

        # Train the best model on the entire training set
        best_model.fit(self.X_train, self.y_train)
        save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        logging.info('Find best model completed')
        return best_model
