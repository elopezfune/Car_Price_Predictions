# Machine Learning ToolBox
# ========================

# Standard libraries
# ==================
import numpy as np
import pandas as pd


# Preprocessing libraries
# =======================
from sklearn.model_selection import train_test_split

# Machine Learning 
# ================
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import r2_score, mean_squared_error

# Optimizers
# ==========
from scipy.optimize import basinhopping


# Neglect warnings
# ================
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)




class Random_Forest_Regresor(RandomForestRegressor):
    def __init__(self,bounds):
        super().__init__()
        if not isinstance(bounds,list):
                raise ValueError('Bounds should be a list of tuples.')
        self.bounds = bounds
        self.optimal = 0.0
    
    # Produces an optimized model
    def fit(self,X,y):
        # Defines the metric to optimize the hyper parameters of the model
        def metric(params):
            # Creates train+validation subsamples
            X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3)
            # Loads the model and make predictions
            model = self.random_forest_model(X_train,y_train,params)
            y_pred = model.predict(X_valid)
            # Returns the root mean square error, which will be minimized later
            return 1.0-r2_score(y_valid, y_pred, multioutput='variance_weighted')
                    
        # Minimizer using the Basin-Hoping algorithm
        x0 = [np.mean(el) for el in self.bounds]
        # Uses the method L-BFGS-B because the problem is smooth and bounded
        minimizer_kwargs = dict(method="L-BFGS-B",bounds=self.bounds)
        opt = basinhopping(metric, x0, minimizer_kwargs=minimizer_kwargs)
        self.optimal = opt.fun
        model = self.random_forest_model(X,y,opt.x)
        return model
    
    # Makes predictions
    def predict(self,X,y=None):
        return self.predict(X)
        
    # Defines the model
    def random_forest_model(self,X_train,y_train,params):
        n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, random_state = params
        model = RandomForestRegressor(n_estimators = int(n_estimators), 
                                      max_features = int(max_features),
                                      max_depth = int(max_depth),
                                      min_samples_split = int(min_samples_split),
                                      min_samples_leaf = int(min_samples_leaf),
                                      random_state=int(random_state),
                                      n_jobs=-1)
        # Fits the model
        model = model.fit(X_train,y_train)
        # Returns the fitted model
        return model


class CatBoost_Regresor(CatBoostRegressor):
    def __init__(self,bounds):
        super().__init__()
        if not isinstance(bounds,list):
                raise ValueError('Bounds should be a list of tuples.')
        self.bounds = bounds
        self.optimal = 0.0
    
    # Produces an optimized model
    def fit(self,X,y):
        # Defines the metric to optimize the hyper parameters of the model
        def metric(params):
            # Creates train+validation subsamples
            X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3)
            # Loads the model and make predictions
            model = self.catboost_model(X_train,y_train,params)
            y_pred = model.predict(X_valid)
            # Returns the root mean square error, which will be minimized later
            return 1.0-r2_score(y_valid, y_pred, multioutput='variance_weighted')
                    
        # Minimizer using the Basin-Hoping algorithm
        x0 = [np.mean(el) for el in self.bounds]
        # Uses the method L-BFGS-B because the problem is smooth and bounded
        minimizer_kwargs = dict(method="L-BFGS-B",bounds=self.bounds)
        opt = basinhopping(metric, x0, minimizer_kwargs=minimizer_kwargs)
        self.optimal = opt.fun
        model = self.catboost_model(X,y,opt.x)
        return model
    
    # Makes predictions
    def predict(self,X,y=None):
        return self.predict(X)
        
    # Defines the model
    def catboost_model(self,X_train,y_train,params):
        n_estimators, learning_rate, max_depth, random_state = params
        model = CatBoostRegressor(n_estimators = int(n_estimators), 
                                  learning_rate = learning_rate,
                                  max_depth = int(max_depth),
                                  random_state=int(random_state),
                                  verbose=False)
        # Fits the model
        model = model.fit(X_train,y_train)
        # Returns the fitted model
        return model




class CatBoost_Classifier(CatBoostRegressor):
    def __init__(self,bounds):
        super().__init__()
        if not isinstance(bounds,list):
                raise ValueError('Bounds should be a list of tuples.')
        self.bounds = bounds
        self.optimal = 0.0
    
    # Produces an optimized model
    def fit(self,X,y):
        # Defines the metric to optimize the hyper parameters of the model
        def metric(params):
            # Creates train+validation subsamples
            X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3)
            # Loads the model and make predictions
            model = self.catboost_model(X_train,y_train,params)
            y_pred = model.predict(X_valid)
            # Returns the root mean square error, which will be minimized later
            return 1.0-r2_score(y_valid, y_pred, multioutput='variance_weighted')
                    
        # Minimizer using the Basin-Hoping algorithm
        x0 = [np.mean(el) for el in self.bounds]
        # Uses the method L-BFGS-B because the problem is smooth and bounded
        minimizer_kwargs = dict(method="L-BFGS-B",bounds=self.bounds)
        opt = basinhopping(metric, x0, minimizer_kwargs=minimizer_kwargs)
        self.optimal = opt.fun
        model = self.catboost_model(X,y,opt.x)
        return model
    
    # Makes predictions
    def predict(self,X,y=None):
        return self.predict(X)
        
    # Defines the model
    def catboost_model(self,X_train,y_train,params):
        n_estimators, learning_rate, max_depth, random_state = params
        model = CatBoostRegressor(n_estimators = int(n_estimators), 
                                  learning_rate = learning_rate,
                                  max_depth = int(max_depth),
                                  random_state=int(random_state),
                                  verbose=False)
        # Fits the model
        model = model.fit(X_train,y_train)
        # Returns the fitted model
        return model





