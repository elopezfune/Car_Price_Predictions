from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import source.config as config
import source.load_push_data as lp
import source.data_analytics as da
from source.preprocessing_functions import Car_Names, Change_Data_Type, Years_Passed, Currency_Corrector
from source.preprocessing_functions import Unit_Corrector, Remove_Substring, Duplicated_Data, Skew_Kurt
from source.preprocessing_functions import One_Hot_Encoding_Train, One_Hot_Encoding_Tests 
from source.preprocessing_functions import Standard_Scaling_Train, Standard_Scaling_Tests, Reduce_Memory_Usage
from source.machine_learning_toolbox import Random_Forest_Regresor, CatBoost_Regresor
import source.predict as pr 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

np.random.seed(seed=config.SEED)

app = Flask(__name__)
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/analytics',methods=['GET','POST'])
def analytics():
    
    # Formatting block
    # ================
    data = lp.load_data(config.DATA)
    data.drop('Torque',axis=1,inplace=True)
    data.dropna(inplace=True)
    brand_branch = Car_Names('Name')
    data = brand_branch.fit_transform(data)
    to_integer = Change_Data_Type('Seats','Integer')
    data = to_integer.fit_transform(data)
    to_string = Change_Data_Type('Seats','String')
    data = to_string.fit_transform(data)
    to_float = Change_Data_Type('Selling_Price','Float')
    data = to_float.fit_transform(data)
    to_float = Change_Data_Type('Kms_Driven','Float')
    data = to_float.fit_transform(data)
    years_passed = Years_Passed('Year',config.YEAR)
    data = years_passed.fit_transform(data)
    corrector = Unit_Corrector('Mileage',1.40)
    data = corrector.fit_transform(data)
    data = data.rename(columns={"Mileage": "Mileage[kmpl]"})
    rupee_euros = Currency_Corrector(config.TARGET,config.CONVERSION_RATE)
    data = rupee_euros.fit_transform(data)
    remove_substring = Remove_Substring('Engine',' CC')
    data = remove_substring.fit_transform(data)
    to_float = Change_Data_Type('Engine','Float')
    data = to_float.fit_transform(data)
    remove_substring = Remove_Substring('Max_Power',' bhp')
    data = remove_substring.fit_transform(data)
    data['Max_Power'] = pd.to_numeric(data['Max_Power'])
    data = to_float.fit_transform(data)
    duplicates = Duplicated_Data()
    data = duplicates.fit_transform(data)
    data[config.TARGET] = 0.011*data[config.TARGET]
    # Formatting block
    # ================
    if request.method == 'POST':
        to_plot = str(request.form['Column'])
        if to_plot in config.CAT_VARS:
            plots = Markup(da.Plot_Categorical_API(data,to_plot,config.TARGET))
        else:
            plots = Markup(da.Plot_Continuous_API(data,to_plot,config.TARGET))            
        return render_template('analytics.html', analytics = plots)
    else:
        return render_template('analytics.html')
    
    #return render_template('analytics.html')


@app.route('/modeling',methods=['GET','POST'])
def modeling():
    # Preprocessing training subsample for modeling
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
    testings = lp.load_data(config.TESTINGS)
    # Label encoding of categorical variables
    # =======================================    
    one_hot_tests = One_Hot_Encoding_Tests(config.CAT_VARS,list(training.columns))
    testings = one_hot_tests.fit_transform(testings)
    # Normalization and scaling
    # =========================
    scale_tests = Standard_Scaling_Tests(config.FLOAT_VARS,scale_train)
    testings = scale_tests.fit_transform(testings)

    columns = training.drop(config.TARGET,axis=1).columns
    
    # Loads model from path
    # =====================
    with open(config.CB_MODEL,'rb') as f:
        model = joblib.load(f)
    
    if request.method == 'POST':
        fim = Markup(da.Feature_Importances_API(model,columns))
        cal = Markup(da.Calibrations_API(model,training,testings,config.TARGET))        
        return render_template('modeling.html', feature_importance = fim, calibrations=cal)
    else:
        return render_template('modeling.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    to_predict = {}
    if request.method == 'POST':
        to_predict['Brand'] = str(request.form['Brand'])
        to_predict['Year'] = float(request.form['Year'])
        to_predict['Kms_Driven'] = float(request.form['Kms_Driven'])
        to_predict['Owner'] = str(request.form['Owner'])
        to_predict['Fuel'] = str(request.form['Fuel'])
        to_predict['Seller_Type'] = str(request.form['Seller_Type'])
        to_predict['Transmission'] = str(request.form['Transmission'])
        to_predict['Seats'] = str(request.form['Seats'])
        to_predict['Mileage[kmpl]'] = float(request.form['Mileage[kmpl]'])
        to_predict['Engine'] = float(request.form['Engine'])
        to_predict['Max_Power'] = float(request.form['Max_Power'])
        result = pr.price_predictions(config.CB_MODEL,to_predict)[0]
        result = round(result,2) # Conversion from Indian Rupee to Euros
        
        if result<=0:
            return render_template('predict.html',prediction_texts="Sorry we cannot sell this car!")
        else:
            return render_template('predict.html',prediction_text="You can buy the {} for {} [EUR]".format(to_predict['Brand'],result))
    else:
        return render_template('predict.html')     
        
        print(to_predict)
        return render_template('predict.html')







if __name__=="__main__":
    app.run(debug=True,port=8000)
