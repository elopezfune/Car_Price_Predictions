# Data manipulation
import numpy as np
import pandas as pd
import source.config as config

# For plotting
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=18) # To use big fonts...
plt.rcParams['figure.figsize'] = (16.0, 10.0) # To set figure sizes for big plots
matplotlib.rc('figure', facecolor='white')  # To set white as the background color for plots
matplotlib.rcParams.update({'font.size': 18})

# For Hypothesis Testing
from scipy.stats import normaltest, shapiro, kstest 

# ===========================================================================================================
# Plot Groups
# ===========================================================================================================
def Plot_Group_By(df,column_group,target):
    # Copies the dataframe
    df = df.copy()
    df.groupby(column_group)[target].mean().sort_values().plot.barh()
    plt.xlabel('Average '+target.replace('_', ' '))
    plt.show()
def Plot_Categorical_API(df,column_group,target):
    # Copies the dataframe
    df = df.copy()
    df_1 = pd.DataFrame(df[column_group].value_counts(ascending=False))
    df_1.rename(columns={column_group:'Count'},inplace=True)
    df_1.index.name=column_group
    df_2 = df = pd.DataFrame(df.groupby(column_group)[target].mean().sort_values(ascending=False))
    df_2.rename(columns={'Selling_Price':'Average Selling Price'},inplace=True)
    df = df_1.join(df_2)
    dict_df = {}
    dict_df[column_group] = list(df.index)
    dict_df['Count'] = list(df.Count.values)
    dict_df['Average Selling Price'] = list(df['Average Selling Price'].values)
    fig = make_subplots(rows=2, cols=1)
    fig.append_trace(go.Bar(name="Most sold", x=dict_df['Count'], y=dict_df[column_group], orientation='h'), row=1, col=1)
    fig.append_trace(go.Bar(name="Average Selling Price", x=dict_df['Average Selling Price'], y=dict_df[column_group], orientation='h'), row=2, col=1)
    fig.update_layout(height=650, width=1200)
    return plotly.offline.plot(fig, include_plotlyjs=True, output_type='div')

def Plot_Continuous_API(df,column_group,target):
    # Copies the dataframe
    df = df.copy()
    df_1 = pd.DataFrame(df.groupby(column_group)[target].mean().sort_index(ascending=True))
    df_1.rename(columns={target:'Average Selling Price'},inplace=True)
    df_1[column_group] = df_1.index
    df_1.reset_index(drop=True,inplace=True)
    fig1 = px.histogram(df, x=column_group, marginal="box")
    fig2 = px.line(df_1, x=column_group, y="Average Selling Price")
    fig2.update_traces(line_color='red')
    fig = make_subplots(rows=2, cols=1, specs=[[{'type': 'box'}], [{'type':'xy', 'secondary_y':True}]])
    fig.add_trace(go.Box(fig1.data[1]), row=1, col=1)
    fig.add_trace(go.Histogram(fig1.data[0]), row=2, col=1)
    fig.add_trace(go.Scatter(fig2.data[0]), secondary_y=True, row=2, col=1)
    fig.update_layout(yaxis=dict(domain=[0.8516, 1.0]),
                      yaxis2=dict(domain=[0.0, 0.8316], title_text='Count'),
                      yaxis3=dict(title_text='Average Selling Price', titlefont=dict(color="red"),
                                  tickfont=dict(color="red")),
                      xaxis=dict(showticklabels=False),
                      xaxis2=dict(title_text=column_group),)
    fig.update_layout(xaxis_title=column_group,height=650, width=1200)
    return plotly.offline.plot(fig, include_plotlyjs=True, output_type='div')    
# ===========================================================================================================
# Plot Groups
# ===========================================================================================================


# ===========================================================================================================
# Normal distribution hypothesis tests
# ===========================================================================================================
def Normality_Check(residuals,threshold):
    # Different normality hypothesis tests
    p_normal = normaltest(residuals).pvalue
    #p_shapiro = shapiro(residuals).pvalue
    p_kstest = kstest(residuals, 'norm').pvalue 
    
    def printing_function(p,alpha):
        if p < alpha:  # Null Hypothesis: x comes from a Normal Distribution
            return "The null hypothesis can be rejected"
        else:
            return "The null hypothesis cannot be rejected"
    
    print('Normal Test:', printing_function(p_normal,threshold))
    #print('Shapiro Test:', printing_function(p_shapiro,threshold))
    print('Kolmogorov Test:', printing_function(p_kstest,threshold))
# ===========================================================================================================
# Normal distribution hypothesis tests
# ===========================================================================================================

# ===========================================================================================================
# Model calibration
# ===========================================================================================================
def Calibration(y_true,y_pred):
    # Sets figure size
    plt.rcParams['figure.figsize'] = (8.0,8.0)  
    # Plots true values versus predicted ones
    plt.scatter(y_true,y_pred)
    r = np.linspace(0, max(y_true), num=100)
    plt.plot(r,r)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.xlim(0,max(y_true)+1)
    plt.ylim(0,max(y_pred)+1)
    plt.show()

def Calibrations_API(model,train,tests,target):
    X_train, X_tests = train.drop(target,axis=1), tests.drop(target,axis=1)
    y_train, y_tests = train[target].values, tests[target].values
    y_train_pred, y_tests_pred = model.predict(X_train), model.predict(X_tests)
    y_train_pred, y_tests_pred = 10.0**(y_train_pred), 10.0**(y_tests_pred)
    
    trace1 = go.Scatter(x=y_train,y=y_train_pred,name='Training subample',mode="markers")
    trace2 = go.Scatter(x=y_tests,y=y_tests_pred,name='Testing subsample',yaxis='y2',mode="markers")
    trace3 = go.Scatter(x=[0,1],y=[0,1],name='Perfect Calibration')
    trace3 = go.Scatter(x=[min(y_train),max(y_train)],y=[min(y_train),max(y_train)],name='Train Perfect Predictions')
    trace4 = go.Scatter(x=[min(y_tests),max(y_tests)],y=[min(y_tests),max(y_tests)],name='Tests Perfect Predictions')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(trace1)
    fig.add_trace(trace3)
    fig.add_trace(trace2,secondary_y=True)
    fig.add_trace(trace4,secondary_y=True)
    fig.update_layout(xaxis_title='Real Values', yaxis_title='Train Predicted Values',
                      yaxis2=dict(domain=[0.0, 0.8316], title_text='Tests Predicted Values'),
                      height = 750, width = 650, xaxis=dict(tickangle=-90))
    return plotly.offline.plot(fig, include_plotlyjs=True, output_type='div') 
# ===========================================================================================================
# Model calibration
# ===========================================================================================================



# ===========================================================================================================
# Feature Importances
# ===========================================================================================================
def Feature_Importances(model,columns):
    # Computes the feature importances
    importances = model.feature_importances_
    # Computes the Standard Errors
    #std_err = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # Creates a dataframe with the compiled data
    feature_importances = pd.DataFrame(index=columns)
    feature_importances.loc[:,'Importances'] = importances
    #feature_importances.loc[:,'Errors'] = std_err
    feature_importances = feature_importances.sort_values('Importances')
    
    # Creates a plot
    fig, ax = plt.subplots()
    #feature_importances['Importances'].plot.barh(xerr=feature_importances.Errors, ax=ax)
    feature_importances['Importances'].plot.barh(ax=ax)
    ax.set_title("Feature importances using Mean Decrease in Impurity")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

def Feature_Importances_API(model,columns):
    # Computes the feature importances
    importances = model.feature_importances_
    # Creates a dataframe with the compiled data
    feature_importances = pd.DataFrame(index=columns)
    #Normalizing feature importances
    feature_importances.loc[:,'Importances'] = importances/max(importances)
    feature_importances = feature_importances.sort_values('Importances')
    feature_importances = feature_importances[feature_importances['Importances']>=0.01]
    
    # Creates a plot with the feature importances
    fig = px.bar(feature_importances, x='Importances', orientation='h')
    fig.update_layout(xaxis_title='Normalized Importances (>1%)', yaxis_title='Features',height=750, width=600)
    return plotly.offline.plot(fig, include_plotlyjs=True, output_type='div') 
# ===========================================================================================================
# Feature Importances
# ===========================================================================================================

