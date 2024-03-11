import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class Predictor:
    def __init__(self):
        pass
    def predict(self, X):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            self.model = load_object(model_path)
            predictions = self.model.predict(X)
            logging.info('Predictions made successfully')
            return predictions
        except Exception as e:
            logging.error(f"Error occurred during prediction: {str(e)}")
            raise CustomException("Error occurred during prediction")

# Example usage:
# predictor = Predictor()
# Assuming df_unseen is the DataFrame containing the unseen data
# predictions = predictor.predict(df_unseen)
# print(predictions)
