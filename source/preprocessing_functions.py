# =======================================================================================================
# DATA PREPROCESSING TOOLBOX
# =======================================================================================================
# -------------------------------------------------------------------------------------------------------

# Standard libraries
# ==================
import numpy as np
import pandas as pd


# Preprocessing libraries
# =======================
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats


# Neglect warnings
# ================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



# =======================================================================================================
# Creates Brand and Branch columns
# =======================================================================================================
class Car_Names(BaseEstimator,TransformerMixin):
    def __init__(self,names):
        if not isinstance(names,str):
            raise ValueError('Names should be a string name of a column.')
        self.names = names
    
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        if self.names not in X.columns:
            raise ValueError('Name should be a column of the dataframe.')
        # Copies the dataframe
        X = X.copy()
        # Extracts the brand names and distribution
        brands = [el.split()[:1][0] for el in X[self.names]]
        # Deletes initial name column
        X.drop(self.names,axis=1,inplace=True)
        # Returns a dataframe with the new columns
        X["Brand"] = brands
        return X
# =======================================================================================================
# Creates Brand and Branch columns
# =======================================================================================================




# =======================================================================================================
# Changes data type of column
# =======================================================================================================
class Change_Data_Type(BaseEstimator, TransformerMixin):
    def __init__(self,column,data_type):
        if not isinstance(column,str):
            raise ValueError('Column should be a string name of a column.')
        if not isinstance(data_type,str):
            raise ValueError('Data_Type should be a string from the list: [String, Integer, Float].')
        self.column = column
        self.data_type = data_type
        
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Copies the dataframe
        X = X.copy()
        # Returns a dataframe with the new column in new data type
        if self.data_type == 'String':
            X[self.column] = X[self.column].apply(lambda x: str(x))
        elif self.data_type == 'Integer':
            X[self.column] = X[self.column].apply(lambda x: int(x))
        elif self.data_type == 'Float':
            X[self.column] = X[self.column].apply(lambda x: float(x))
        else:
            raise ValueError('Data_Type should be a string from the list: [String, Integer, Float].')
        return X
# =======================================================================================================
# Changes data type of column
# =======================================================================================================



# =======================================================================================================
# Creates Years passed variable
# =======================================================================================================
class Years_Passed(BaseEstimator, TransformerMixin):
    def __init__(self,year_column,actual_year):
        if not isinstance(year_column,str):
            raise ValueError('Year_Column should be a string name of a column.')
        if not isinstance(actual_year,int):
            raise ValueError('Actual_Year should be an integer number.')
        self.year_column = year_column
        self.actual_year = actual_year
        
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        # Copies the dataframe
        X = X.copy()
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Returns a dataframe with the new column of years passed
        X[self.year_column] = X[self.year_column].apply(lambda x: float(self.actual_year-x))
        return X
# =======================================================================================================
# Creates Years passed variable
# =======================================================================================================



# =======================================================================================================
# KM/KG to KMPL Units corrector
# =======================================================================================================
class Unit_Corrector(BaseEstimator, TransformerMixin):
    def __init__(self,column,factor):
        if not isinstance(column,str):
            raise ValueError('Column should be a string name of a column.')
        if not isinstance(factor,float):
            raise ValueError('Factor should be a float conversion factor.')
        self.column = column
        self.factor = factor
        
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        # Copies the dataframe
        X = X.copy()
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Conversion step
        result = []
        for el in X[self.column]:
            if str(el).endswith(' km/kg'):
                el = el.replace(' km/kg','')
                el = float(el)*self.factor
                result.append(float(el))
            elif str(el).endswith(' kmpl'):
                el = el.replace(' kmpl','')
                result.append(float(el))
        X[self.column] = result
        # Returns a dataframe with the new column converted to the KMPL units        
        return X
# =======================================================================================================
# KM/KG to KMPL Units corrector
# =======================================================================================================



# =======================================================================================================
# Currency units corrector
# =======================================================================================================
class Currency_Corrector(BaseEstimator, TransformerMixin):
    def __init__(self,column,factor):
        if not isinstance(column,str):
            raise ValueError('Column should be a string name of a column.')
        if not isinstance(factor,float):
            raise ValueError('Factor should be a float conversion factor.')
        self.column = column
        self.factor = factor
        
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        # Copies the dataframe
        X = X.copy()
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Conversion step
        X[self.column] = X[self.column].apply(lambda x: self.factor*x)
        # Returns a dataframe with the new column converted to the KMPL units        
        return X
# =======================================================================================================
# Currency units corrector
# =======================================================================================================



# =======================================================================================================
# Removes substrings from columns
# =======================================================================================================
class Remove_Substring(BaseEstimator, TransformerMixin):
    def __init__(self,column,substring):
        if not isinstance(column,str):
            raise ValueError('Column should be a string name of a column.')
        if not isinstance(substring,str):
            raise ValueError('Substring should be an integer number.')
        self.column = column
        self.substring = substring
        
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        # Copies the dataframe
        X = X.copy()
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Returns a dataframe with the new column without the substring
        X[self.column] = X[self.column].apply(lambda x: x.replace(self.substring,''))
        return X
# =======================================================================================================
# Removes substrings from columns
# =======================================================================================================





# =======================================================================================================
# Checks for duplicated data and removes them
# =======================================================================================================
class Duplicated_Data(BaseEstimator, TransformerMixin):
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        # Copies the dataframe
        X = X.copy()
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Returns a dataframe with the duplicated rows removed
        return X.drop_duplicates()
# =======================================================================================================
# Checks for duplicated data and removes them
# =======================================================================================================



# =======================================================================================================
# Skewness and Kurtosis
# =======================================================================================================
class Skew_Kurt(BaseEstimator, TransformerMixin):
    def __init__(self,num_vars):
        if not all(isinstance(el, str) for el in num_vars):
            raise ValueError('Target should be a string names.')
        self.num_vars = num_vars
    def fit(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        return self
    def predict(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        #if self.num_vars not in X.columns:
        #    raise ValueError('Num_vars should be columns of the dataframe.')
        else:
            result = pd.DataFrame(columns=['Skewness','Kurtosis'],index=self.num_vars)
            result.loc[:,'Skewness'] = [X[el].skew() for el in self.num_vars]
            result.loc[:,'Kurtosis'] = [X[el].kurt() for el in self.num_vars]
            return result
# =======================================================================================================
# Skewness and Kurtosis
# =======================================================================================================


# =======================================================================================================
# One Hot Encoding
# =======================================================================================================
class One_Hot_Encoding_Train(BaseEstimator, TransformerMixin):
    
    def __init__(self,cat_vars,drop_first=False):
        if not isinstance(cat_vars,list):
            raise ValueError('Cat_vars should be a list of strings.')
        elif not isinstance(drop_first,bool):
            raise ValueError('Drop_first should be boolean.')
        self.cat_vars = cat_vars
        self.drop_first = drop_first
        self.encoded_columns = []
    
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self


    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Copies the fataframe
        X = X.copy()
        # Encodes the categorical
        df_to_encode = X[self.cat_vars]
        df_encoded = pd.get_dummies(df_to_encode,drop_first=self.drop_first)
        #Formats the names of the variables
        df_encoded.columns = [el.replace('_','[')+']' for el in df_encoded.columns]
        X = pd.concat([X,df_encoded],axis=1).drop(self.cat_vars,axis=1)
        del df_to_encode, df_encoded
        self.encoded_columns = X.columns
        # Returns the dataframe with the one-hot-encoded categorical variables
        return X

    
class One_Hot_Encoding_Tests(BaseEstimator, TransformerMixin):
    
    def __init__(self,cat_vars,train_cols,drop_first=False):
        if not isinstance(cat_vars,list):
            raise ValueError('Cat_vars should be a list of strings.')
        elif not isinstance(drop_first,bool):
            raise ValueError('Drop_first should be boolean.')
        if not isinstance(train_cols,list):
            raise ValueError('Train_Cols should be a list of strings.')
        self.cat_vars = cat_vars
        self.drop_first = drop_first
        self.train_cols = train_cols
    
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self


    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Copies the fataframe
        X = X.copy()
        # Encodes the categorical
        df_to_encode = X[self.cat_vars]
        df_encoded = pd.get_dummies(df_to_encode,drop_first=self.drop_first)
        #Formats the names of the variables
        df_encoded.columns = [el.replace('_','[')+']' for el in df_encoded.columns]
        X = pd.concat([X,df_encoded],axis=1).drop(self.cat_vars,axis=1)
        del df_to_encode, df_encoded
        
        # Returns the dataframe with the one-hot-encoded categorical variables
        if set(X.columns) == set(self.train_cols):
            return X
        else:
            # Checks for extra columns in the train or the test subsamples
            tests_train = list(set(X.columns) - set(self.train_cols))
            train_tests = list(set(self.train_cols) - set(X.columns))
            if len(tests_train) != 0:
                X.drop(tests_train,axis=1,inplace=True)
            if len(train_tests) != 0:
                for el in train_tests:
                    X[el] = [0]*len(X)
            return X    
# =======================================================================================================
# One Hot Encoding
# =======================================================================================================


# =======================================================================================================
# Scaling numerical variables
# =======================================================================================================

class Standard_Scaling_Train(BaseEstimator, TransformerMixin):
    
    def __init__(self,num_vars):
        if not isinstance(num_vars,list):
            raise ValueError('Num_vars should be a list of strings.')
        self.num_vars = num_vars
        self.scaler = None
        
    
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Copies the fataframe
        X = X.copy()
        df_scale = X[self.num_vars]
        
        # Trains the scaler
        sc_X = StandardScaler().fit(df_scale)
        # Saves the model
        self.scaler = sc_X
        # Scales the variables
        X[df_scale.columns] = sc_X.transform(df_scale)
        # Returns the dataframe with the variables scaled
        return X

    
class Standard_Scaling_Tests(BaseEstimator, TransformerMixin):
    
    def __init__(self,num_vars,scaler):
        if not isinstance(num_vars,list):
            raise ValueError('Num_vars should be a list of strings.')
        self.num_vars = num_vars
        self.scaler = scaler
            
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Copies the fataframe
        X = X.copy()
        df_scale = X[self.num_vars]
        # Loads the trained scaler
        sc_X = self.scaler
        # Scales the variables
        X[df_scale.columns] = sc_X.transform(df_scale)
        # Returns the dataframe with the variables scaled
        return X
# =======================================================================================================
# Scaling numerical variables
# =======================================================================================================


# =======================================================================================================
# Outliers detector and removal
# =======================================================================================================
class Outliers_Removal(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.outliers = pd.DataFrame()
    
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Copies the fataframe
        X = X.copy()
        # Computes the first and third quantiles
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        #Computes the inter-quantile range
        IQR = Q3 - Q1
        # Removes all rows that contain outliers
        mask = ((X < (Q1-1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))
        self.outliers = X[mask.any(axis=1)]
        X = X[~mask.any(axis=1)]        
        # Returns a dataframe with the outliers removed
        return X
# =======================================================================================================
# Outliers detector and removal
# =======================================================================================================


# =======================================================================================================
# Removes unuseful predictors
# =======================================================================================================
class Remove_Predictors(BaseEstimator, TransformerMixin):
    def __init__(self,predictors):
        if not isinstance(predictors,list):
            raise ValueError('Predictors should be a list of strings.')        
        self.predictors = predictors
    
    # Fits the model
    def fit(self, X, y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # This step is needed to fit the sklearn Pipeline
        return self
    
    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        X.drop(self.predictors,axis=1,inplace=True)
        return X
# =======================================================================================================
# Removes unuseful predictors
# =======================================================================================================



# =======================================================================================================
# Reduces Memory usage
# =======================================================================================================
class Reduce_Memory_Usage(BaseEstimator, TransformerMixin):
    
    def _init_(self):
        self.initial_memory = 0.0
        self.final_memory = 0.0
        self.reduced_percent = 0.0    
    
    # Fits the model
    def fit(self, X, y=None):
        # This step is needed to fit the sklearn Pipeline
        return self

    def transform(self,X,y=None):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('X should be a dataframe.')
        # Copies the dataframe
        X = X.copy()
        # Counts the memory usage in X
        self.initial_memory = X.memory_usage().sum()/1024**2
        # Loops through all the columns of X to change data-types
        for col in X.columns:
            col_type = X[col].dtype
            if col_type != object:
                c_min = X[col].min()
                c_max = X[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        X[col] = X[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        X[col] = X[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        X[col] = X[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        X[col] = X[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        X[col] = X[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        X[col] = X[col].astype(np.float32)
                    else:
                        X[col] = X[col].astype(np.float64)
            else:
                X[col] = X[col].astype('category')        
        # Counts the memory usage in X after transforming
        self.final_memory = X.memory_usage().sum() / 1024**2
        # Memory usage of dataframe decreased by:
        self.reduced_percent = 100 * (self.initial_memory - self.final_memory) / self.initial_memory
        # Returns the reduced memory dataframe
        return X
# =======================================================================================================
# Reduces Memory usage
# =======================================================================================================






# -------------------------------------------------------------------------------------------------------
# =======================================================================================================
# DATA PREPROCESSING TOOLBOX
# =======================================================================================================