import sys
from dataclasses import dataclass
import pandas as pd
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    # def get_data_transformer_object(self):
    #     '''
    #     This function is responsible for creating a data transformation pipeline.
    #     It handles both numerical and categorical features by applying appropriate transformations.
    #     '''
    #     try:
    #         numerical_features = ['writing score', 'reading score']
    #         categorical_features = [
    #             'gender',
    #             'race/ethnicity',
    #             'parental level of education',
    #             'lunch',
    #             'test preparation course',
    #             ]
    #         num_pipeline=Pipeline(
    #             steps=[
    #                 ("imputer",SimpleImputer(strategy='median')),
    #                 ("scaler",StandardScaler())
    #             ]
    #         )
    #         cat_pipeline=Pipeline(
    #             steps=[
    #                 ('imputer',SimpleImputer(strategy='most_frequent')),
    #                 ('one_hot_encoder',OneHotEncoder()),
    #                 ('scaler',StandardScaler())
    #             ]
    #         )
    #         logging.info(f"Numerical features: {numerical_features}")
    #         logging.info(f"Categorical features: {categorical_features}")
    #         preprocessor=ColumnTransformer([
    #             ('num_pipeline',num_pipeline,numerical_features),
    #             ('cat_pipeline',cat_pipeline,categorical_features)
    #         ])
    #         return preprocessor
    #     except Exception as e:
    #         raise CustomException(e, sys)
    def get_data_transformer_object(self):
        '''
        This function is responsible for creating a data transformation pipeline.
        It handles both numerical and categorical features by applying appropriate transformations.
        '''
        try:
            numerical_features = ['writing score', 'reading score']
            categorical_features = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course',
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    # Remove StandardScaler for categorical features or use with_mean=False
                    ('scaler', StandardScaler(with_mean=False))  # This is the key change
                ]
            )
            
            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function is responsible for initiating the data transformation process.
        It reads the training and testing data, applies the transformations, and saves the preprocessor object.
        '''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame shape: {train_df.shape}")
            logging.info(f"Test DataFrame shape: {test_df.shape}")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name='math score'
            numerical_features=['writing score', 'reading score']
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing object on training and testing dataframes completed")
            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e, sys)