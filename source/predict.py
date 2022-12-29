import numpy as np
import pandas as pd
import source.config as config
import source.load_push_data as lp
from source.preprocessing_functions import Car_Names, Change_Data_Type, Years_Passed
from source.preprocessing_functions import Unit_Corrector, Remove_Substring, Duplicated_Data, Skew_Kurt
from source.preprocessing_functions import One_Hot_Encoding_Train, One_Hot_Encoding_Tests 
from source.preprocessing_functions import Standard_Scaling_Train, Standard_Scaling_Tests, Reduce_Memory_Usage
import joblib

np.random.seed(seed=config.SEED)

# ===================================================================================
# MAKE PREDICTIONS
# ===================================================================================
# -----------------------------------------------------------------------------------

def price_predictions(path_to_model,data_point:dict):
    
    # Preprocessing training subsample for predictions
    # ================================================
    # Loads the data
    # ==============
    training = lp.load_data(config.TRAINING)
    # Drops missing values (NaNs)
    # =================================
    training = training.dropna()
    # Label encoding of categorical variables
    # =======================================
    one_hot_train = One_Hot_Encoding_Train(config.CAT_VARS)
    training = one_hot_train.fit_transform(training)
    # Normalization and scaling
    # =========================
    scale_train = Standard_Scaling_Train(config.FLOAT_VARS)
    training = scale_train.fit_transform(training)
    
    
    # Converts data point to dataframe
    # ================================   
    testings = pd.DataFrame(data_point,index=[0])
    # Label encoding of categorical variables
    # =======================================    
    one_hot_tests = One_Hot_Encoding_Tests(config.CAT_VARS,list(training.columns))
    testings = one_hot_tests.fit_transform(testings)
    # Normalization and scaling
    # =========================
    scale_tests = Standard_Scaling_Tests(config.FLOAT_VARS,scale_train)
    testings = scale_tests.fit_transform(testings)
    
    # Loads model from path
    # =====================
    with open(path_to_model,'rb') as f:
        model = joblib.load(f)
    
    # Make predictions
    # ================
    result = model.predict(testings)
    result = 10.0**result
    # Returns the prediction
    # ======================
    return result
# -----------------------------------------------------------------------------------
# ===================================================================================
# MAKE PREDICTIONS
# ===================================================================================