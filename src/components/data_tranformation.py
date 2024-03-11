import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object



@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()


    def initiate_data_transformation(self,data_path):
        try:
            df = pd.read_csv(data_path)
            logging.info('Read csv already')
            num_features = df.select_dtypes(exclude='object').columns
            df_model = df.copy()
            df_model['average_score'] = df[num_features].sum(axis=1)/len(num_features)
            select_columns = ['math_score','reading_score','writing_score']
            df_model.drop(select_columns,axis=1,inplace=True)
            X = df_model.drop(columns=['average_score'],axis = 1)
            y = df_model['average_score']
            num_features = X.select_dtypes(exclude="object").columns
            cat_features = X.select_dtypes(include="object").columns
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(sparse=False)
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, cat_features),
                    ("StandardScaler", numeric_transformer, num_features),        
                ]
            )

            # Fit and transform the data
            X_encoded = preprocessor.fit_transform(X)
            logging.info('Encoded categorical variable already')
            # Fit the OneHotEncoder
            oh_transformer.fit(X[cat_features])

            # Get column names for one-hot encoded features
            encoded_column_names = oh_transformer.get_feature_names_out(cat_features)

            # Combine column names of one-hot encoded and numerical features
            final_column_names = list(encoded_column_names) + list(X.select_dtypes(exclude="object").columns)
            logging.info('Getting columns name by use value of categorical')
            # Convert the transformed array to a DataFrame with column names
            X_encoded_df = pd.DataFrame(X_encoded, columns=final_column_names)
            logging.info('Get dataframe for train_test_split ')
            X_train, X_test, y_train, y_test = train_test_split(X_encoded_df,y,test_size=0.2,random_state=42)
            logging.info('Tranform completed')
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj = X_encoded_df
            )
            return (
                X_train,
                X_test,
                y_train,
                y_test
                )

        except Exception as e:
            raise CustomException(e,sys)
