#!/usr/bin/env python
# coding: utf-8

# ## Does weather play a factor in when deliveries occur?

# In[ ]:


#Import Libraries
import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,LogisticRegressionCV,LassoCV,ElasticNet,ElasticNetCV,RidgeCV,RidgeClassifierCV,ridge_regression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score,cross_val_predict,RepeatedStratifiedKFold,RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
#need "pip install scikit-optimize"
from skopt.searchcv import BayesSearchCV
from skopt.utils import use_named_args
from skopt import gp_minimize
from timeit import default_timer as timer
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#functions to perform graphing
def graph_it(y_true,y_pred,title="Graph",RQ=1):
# do the different graphing
    plt.rcParams.update({'font.sans-serif':'Arial'})

    if (RQ == 1):
        lables = np.array(False,True)
    else: labels = np.array(['More_Activity','Same_Activity','Less_Activity'])
    
    #confusion matrix
    cm = confusion_matrix(y_true,y_pred,labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(cmap='Greys',colorbar=False)
    plt.title(title)            
    plt.show()

def graph_feature(names,fi,graph_title,thresh=5,tree=True,add_label=True):
    sort_key = fi.argsort()
    plt.figure(figsize=(5,10))
    bars = plt.barh(names[sort_key],fi[sort_key],color='lightgrey',edgecolor='black')
    plt.title(graph_title)
    
    if (add_label==True and tree==True):
        # Add annotation to top 5 bars
        plt.xlabel('Feature Importance')        
        full_count=len(bars)
        exit_count=full_count
        for bar in bars:
            if(exit_count > 5):
                exit_count = exit_count -1
                continue
            else:
                width = bar.get_width()
                label_y = bar.get_y() + bar.get_height()/4
                plt.text(.01, label_y, s=f'{width:.4f}',fontweight='bold',color='black')
                exit_count = exit_count - 1
    elif (add_label==True and tree==False):
        # Add annotation to top and bottom 3 bars
        plt.xlabel('Coefficients') 
        full_count=len(bars)
        exit_count=full_count
        for bar in bars:
            if(exit_count > thresh and exit_count <= full_count-thresh ):
                exit_count = exit_count -1
                continue
            else:
                width = bar.get_width()
                if (width > 0):
                    plot_width = width-width+width/250
                else:plot_width = width-width+width/1000
                label_y = bar.get_y() + bar.get_height() /4
                plt.text(plot_width, label_y, s=f'{width:.4f}',fontweight='bold',color='black')
                exit_count = exit_count - 1    
           
    plt.show()


# In[ ]:


#load dataset into a dataframe and confirm values
full_start = timer()
df_raw = pd.read_csv('DataSets\\Savage_Daily_Ticket_Count_Weather_Export.csv')
df_raw.head(12)


# Response varaibles are:  
# delivery_count_sum -> total deliveries for the date
# 
# log_ratio_to_average -> ratio of today's deliveries to the average on that day for the last 11 years (log transformed to normallize)
# 
# is_above_average_delivery_day -> to try logistic regression:  True if greater than the average

# In[ ]:


#establish the working set
df = df_raw.copy()

#don't need the delivery date field for analysis nor the day name
df = df.drop(['delivery_date','delivery_weekday_name'],axis=1)

#set values to string for analysis
df['SnowOnGround'] = df['SnowOnGround'].astype(str)
df['is_midweek'] = df['is_midweek'].astype(str)


# In[ ]:


#Correlations have been checked in dataset building, but will put a matrix up for completeness
corr = df_raw.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:


#function to handle multi-collinearity tests
def vif_calc(X):
    vif_info = pd.DataFrame()
    vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_info['Column'] = X.columns
    vif_info.sort_values('VIF', ascending=False)
    return(vif_info)

#function to pass back AIC for linear model

def aic_calc(X,Y):
    #add constant to predictor variables
    X = sm.add_constant(X)
    #fit regression model
    model = sm.OLS(Y, X).fit()
    return(model.aic)


# In[ ]:


#create data sets for analysis, keep x's separate in case I change individual tests first is regression against the delivery count
xc_full = df.drop(['delivery_count_sum','is_above_average_delivery_day','log_ratio_to_average'],axis=1)
xc_full = pd.get_dummies(xc_full,drop_first = True)                #make dummies for categorical values for analysis
yc = df['delivery_count_sum']


# In[ ]:


vif_calc(xc_full)


# This data (as expected) has columns that have correlations.  In general the "Diff" fields are better distributed than the "Amt" so we will remove and run this again.  Also remove the "no precip" fields since they have high correlations.

# In[ ]:


temp_xc = xc_full.drop(['PriorDayPrecipitationAmt','Prior2DayPrecipitationAmt','PriorDaySnowfallAmt','Precip_12AM-6AM_No Precip','Precip_6AM-10AM_No Precip','Precip_6AM-12PM_No Precip','Precip_10AM-2PM_No Precip'],axis=1)
vif_calc(temp_xc)


# All looking good except need to pick which Temperature Diff field to keep. Check for lowest AIC from keeping 1 of them.

# In[ ]:


aic_val =np.empty(4)
aic_val[0] = aic_calc(temp_xc,yc)
temp_x = temp_xc.drop(['PriorDayMaximumTemperatureDiff','PriorDayMinimumTemperatureDiff'],axis=1)
aic_val[1] = aic_calc(temp_x,yc)
temp_x = temp_xc.drop(['PriorDayMinimumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
aic_val[2] = aic_calc(temp_x,yc)
temp_x = temp_xc.drop(['PriorDayMaximumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
aic_val[3] = aic_calc(temp_x,yc)
print(aic_val)


# In[ ]:


xc_part = temp_xc.drop(['PriorDayMaximumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
vif_calc(xc_part)


# I now have a full datasets and a subset of data that has strong VIF stats.  I will run linear regression to get a baseline

# In[ ]:


#function to run baseline regression
def get_stats(x,y,log=False):
    if (log == True):
        results = sm.Logit(y,x).fit()
    else: results = sm.OLS(y,x).fit()
    print(results.summary())


# In[ ]:


#do linear regression on the full (for comparison) and then on the partial removing non-signficant p-values
#real models will have test/train data
get_stats(xc_full,yc)

get_stats(xc_part,yc) 
temp_xc = xc_part.drop(['PriorDaySnowfallDiff','Precip_10AM-2PM_Precip'],axis=1)
get_stats(temp_xc,yc) 

temp_xc = temp_xc.drop(['DailySnowDepth','DailySnowfall','Precip_6AM-10AM_Precip','Precip_6AM-12PM_Precip'],axis=1)
get_stats(temp_xc,yc) 

#capture the dataset used by linear regression
xc_sp = temp_xc


# set up models:  
# 
# Lasso & Elastic Net and "boosting" for of decisions trees for the full models (since they do various measures of feature selection) .
# 
# Ridge Regression and Random Forests for the partial datasets since they work better on models that don't have collinearity issues (which we've addressed).
# 
# In all cases, I will utilize cross validation on all the data to find the "Best" model, and then use a train/test set to get the  root mean squared error and R^2 values.

# In[ ]:


def run_linear(X,Y,graph=False,graph_title='Regression Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'fit_intercept' : [True,False],
        'positive' : [True,False]
    }    
    cv = KFold(n_splits = 10,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=LinearRegression(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=10,
        #scoring="accuracy",  -- leave as default which is based on the estimator
        verbose=0,
        random_state=5440
    )
    #scale the x predictor values and then run the Bayesian search and capture best parameters
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)       
    search.fit(x_scaled,Y)    
    best_params = search.best_params_
    print(best_params)
    
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,Y,test_size=.24,random_state=5440)   

    model = LinearRegression(n_jobs=-1,fit_intercept=best_params['fit_intercept'],positive=best_params['positive'])
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
    r2_test = r2_score(y_test, pred_test)  
    
    if graph:
        graph_feature(X.columns,model.coef_,graph_title,tree=False)
    
    return(rmse_test,r2_test)
    
# function for fitting trees of various depths for Random Forests
def run_cross_validation_on_regression_RF(X, Y,graph=False,graph_title='Regression Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_depth': (1, 9),
        'criterion': ['squared_error','poisson'], #can use poisson if not negative
        'max_features' : [.250,.3333,.375]
    }
    cv = KFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=RandomForestRegressor(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=125,
        #scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    search.fit(X,Y)    
    best_params = search.best_params_
    print(best_params)
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.24,random_state=5440)
    model = RandomForestRegressor(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth']
                                   ,criterion=best_params['criterion'],max_features=best_params['max_features'])
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)    
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
    r2_test = r2_score(y_test, pred_test)  
    
    if graph:
        graph_feature(X.columns,model.feature_importances_,graph_title)
    
    return(rmse_test,r2_test)

# function for fitting trees of various depths for Boosted Version
def run_cross_validation_on_regression_Boost(X, Y,graph=False,graph_title='Regression Graph'):
    #X = predictors, Y = response
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'n_estimators': [500, 600, 700, 800, 900, 1000],
        'max_depth': (1, 9),
        'criterion': ['friedman_mse','squared_error'],
        'loss' : ['squared_error','absolute_error','huber'],
        'max_features' : [.250,.3333,.375]
    }    
    cv = KFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=GradientBoostingRegressor(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=150,
        #scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    search.fit(X,Y)    
    best_params = search.best_params_
    print(best_params)
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.24,random_state=5440)
    model = GradientBoostingRegressor(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth']
                                   ,criterion=best_params['criterion'],loss=best_params['loss'],max_features=best_params['max_features'])
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
    r2_test = r2_score(y_test, pred_test)  
    
    if graph:
        graph_feature(X.columns,model.feature_importances_,graph_title)
    
    return(rmse_test,r2_test)

#enet regression:  handles E-Net and Lasso
def run_enet_regression(X,Y,graph=False,graph_title='Regression Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'alpha' : [.0001,.0005,.001,.005,.01,.05,.1,.5,1.0,5,10,50,100,500,1000],
        'l1_ratio' : [.01,.05,.1,.3,.5,.7,.9,.95,.99,1],
        'fit_intercept' : [True,False]
    }    
    cv = KFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=ElasticNet(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=200,
        #scoring="accuracy",  -- leave as default which is based on the estimator
        verbose=0,
        random_state=5440
    )
    #scale the x predictor values and then run the Bayesian search and capture best parameters
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)       
    search.fit(x_scaled,Y)    
    best_params = search.best_params_
    print(best_params)
    
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,Y,test_size=.24,random_state=5440)   

    model = ElasticNet(fit_intercept=best_params['fit_intercept'],alpha=best_params['alpha'],
                         l1_ratio=best_params['l1_ratio'],random_state=5440)
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
    r2_test = r2_score(y_test, pred_test)  
    
    if graph:
        graph_feature(X.columns,model.coef_,graph_title,tree=False)
    
    return(rmse_test,r2_test)

#ridge regression:  handles Ridge separately due to different hyperparameters and lack of feature selection
def run_ridge_regression(X,Y,graph=False,graph_title='Regression Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression

    #scale the x predictor values and then run the Bayesian search and capture best parameters
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)       
   
    cv = RepeatedKFold(n_splits = 5,n_repeats=75,random_state=5440)  
    model = RidgeCV(alphas=[.0001,.0005,.001,.005,.01,.05,.1,.5,1.0,5,10,50,100,500,1000],cv=cv)
    model.fit(x_scaled,Y)
    print(model.alpha_)
    
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,Y,test_size=.24,random_state=5440)   

    pred_test = model.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
    r2_test = r2_score(y_test, pred_test)  
    
    if graph:
        graph_feature(X.columns,model.coef_,graph_title,tree=False)
    
    return(rmse_test,r2_test)


# In[ ]:


# With the regression functions defined, run the regressions and capture the RMSE and R-squared
start = timer()
linear_rmse,linear_r2 = run_linear(xc_part,yc,graph=True,graph_title="Weather Only - Linear Regression - Partial")
end = timer()
print(f'Linear Model on Data Subset Complete in {end-start} seconds')

start = timer()
linear_sp_rmse,linear_sp_r2 = run_linear(xc_sp,yc,graph=True,graph_title="Weather Only - Linear Regression - Sig P")
end = timer()
print(f'Linear Model on Significant P Values Complete in {end-start} seconds')

start = timer()      
enet_full_rmse,enet_full_r2 = run_enet_regression(xc_full,yc,graph=True,graph_title="Weather Only - Lasso/ENet Regression - Full")
end = timer()
print(f'Enet Regression Model on Full Dataset Complete in {end-start} seconds')

start = timer()  
ridge_part_rmse,ridge_part_r2 = run_ridge_regression(xc_part,yc,graph=True,graph_title="Weather Only - Ridge Regression - Partial")
end = timer()
print(f'Ridge Regression Model on Data Subset Complete in {end-start} seconds')

start = timer()  
rfr_rmse,rfr_r2 = run_cross_validation_on_regression_RF(xc_part,yc,graph=True,graph_title="Weather Only - Random Forests - Partial")
end = timer()
print(f'Random Forest Model on Data Subset Complete in {end-start} seconds')

start = timer()  
boost_rmse,boost_r2 = run_cross_validation_on_regression_Boost(xc_full,yc,graph=True,graph_title="Weather Only - Boosted Trees - Full")
end = timer()
print(f'Boosted Trees Model on Full Dataset Complete in {end-start} seconds')

start = timer()  
boost_part_rmse,boost_part_r2 = run_cross_validation_on_regression_Boost(xc_part,yc,graph=True,graph_title="Weather Only - Boosted Trees - Partial")
end = timer()
print(f'Boosted Trees Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#create a data frame of the results for analysis
result_ds_list  = [['Linear Run 1','Subset',linear_rmse,linear_r2]
                  ,['Linear Run 2','SigP',linear_sp_rmse,linear_sp_r2]
                  ,['ENet Run 1','Full',enet_full_rmse,enet_full_r2]
                  ,['Ridge Run 1','Subset',ridge_part_rmse,ridge_part_r2]
                  ,['Random Forest Run 1','Subset',rfr_rmse,rfr_r2]
                  ,['Boosted Trees Run 1','Full',boost_rmse,boost_r2]
                  ,['Boosted Trees Run 2','Subset',boost_part_rmse,boost_part_r2]]
results_delivery_count = pd.DataFrame(result_ds_list,columns=['Model','Dataset','RMSE','R^2'])
sort_results = results_delivery_count.sort_values(['R^2','RMSE'],ascending=[False,True])
sort_results.to_excel('RQ1_Regression_Count.xlsx')


# There doesn't look to be any predictive power in looking at weather against the daily delivery count.  Next we will look at the ratio of that days loads above average.  I will use the same methods to develop the datasets and run the same tests.  

# In[ ]:


xr_full = df.drop(['delivery_count_sum','is_above_average_delivery_day','log_ratio_to_average'],axis=1)
xr_full = pd.get_dummies(xr_full,drop_first = True)                #make dummies for categorical values for analysis
yr = df['log_ratio_to_average']
xr_part = xr_full.drop(['PriorDayPrecipitationAmt','Prior2DayPrecipitationAmt','PriorDaySnowfallAmt','Precip_12AM-6AM_No Precip','Precip_6AM-10AM_No Precip','Precip_6AM-12PM_No Precip','Precip_10AM-2PM_No Precip'],axis=1)
vif_calc(xr_part)

aic_val =np.empty(4)
aic_val[0] = aic_calc(xr_part,yr)
temp_x = xr_part.drop(['PriorDayMaximumTemperatureDiff','PriorDayMinimumTemperatureDiff'],axis=1)
aic_val[1] = aic_calc(temp_x,yr)
temp_x = xr_part.drop(['PriorDayMinimumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
aic_val[2] = aic_calc(temp_x,yr)
temp_x = xr_part.drop(['PriorDayMaximumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
aic_val[3] = aic_calc(temp_x,yr)
print(aic_val)

xr_part = xr_part.drop(['PriorDayMinimumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
vif_calc(xr_part)


# In[ ]:


#get the significant variables for linear regression and find the proper tuning parametners for Lasso/E-Net
get_stats(xr_part,yr) 
temp_xr = xr_part.drop(['PriorDayMaximumTemperatureDiff','Precip_12AM-6AM_Precip','Precip_6AM-10AM_Precip','Precip_6AM-12PM_Precip','Precip_10AM-2PM_Precip'],axis=1)
get_stats(temp_xr,yr)
xr_sp = temp_xr


# In[ ]:


# With the regression functions defined, run the regressions and capture the RMSE and R-squared
start = timer()
linear_rmse,linear_r2 = run_linear(xr_part,yr,graph=True,graph_title="Weather Only - Linear Regression 2 - Partial")
end = timer()
print(f'Linear Model on Data Subset Complete in {end-start} seconds')

start = timer()
linear_sp_rmse,linear_sp_r2 = run_linear(xr_sp,yr,graph=True,graph_title="Weather Only - Linear Regression 2 - Sig P")
end = timer()
print(f'Linear Model on Significant P Values Complete in {end-start} seconds')

start = timer()      
enet_full_rmse,enet_full_r2 = run_enet_regression(xr_full,yr,graph=True,graph_title="Weather Only - Lasso/ENet Regression 2 - Full")
end = timer()
print(f'Enet Regression Model on Full Dataset Complete in {end-start} seconds')

start = timer()  
ridge_part_rmse,ridge_part_r2 = run_ridge_regression(xr_part,yr,graph=True,graph_title="Weather Only - Ridge Regression 2 - Partial")
end = timer()
print(f'Ridge Regression Model on Data Subset Complete in {end-start} seconds')

start = timer()  
rfr_rmse,rfr_r2 = run_cross_validation_on_regression_RF(xr_part,yr,graph=True,graph_title="Weather Only - Random Forests 2 - Partial")
end = timer()
print(f'Random Forest Model on Data Subset Complete in {end-start} seconds')

start = timer()  
boost_rmse,boost_r2 = run_cross_validation_on_regression_Boost(xr_full,yr,graph=True,graph_title="Weather Only - Boosted Trees 2 - Full")
end = timer()
print(f'Boosted Trees Model on Full Dataset Complete in {end-start} seconds')

start = timer()  
boost_part_rmse,boost_part_r2 = run_cross_validation_on_regression_Boost(xr_part,yr,graph=True,graph_title="Weather Only - Boosted Trees 2 - Partial")
end = timer()
print(f'Boosted Trees Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#create a data frame of the results for analysis
result_ra_list  = [['Linear Run 1','Subset',linear_rmse,linear_r2]
                  ,['Linear Run 2','SigP',linear_sp_rmse,linear_sp_r2]
                  ,['Enet Run 1','Full',enet_full_rmse,enet_full_r2]
                  ,['Ridge Run 1','Subset',ridge_part_rmse,ridge_part_r2]
                  ,['Random Forest Run 1','Subset',rfr_rmse,rfr_r2]
                  ,['Boosted Trees Run 1','Full',boost_rmse,boost_r2]
                  ,['Boosted Trees Run 2','Subset',boost_part_rmse,boost_part_r2]]
results_delivery_count = pd.DataFrame(result_ra_list,columns=['Model','Dataset','RMSE','R^2'])
sort_results = results_delivery_count.sort_values(['R^2','RMSE'],ascending=[False,True])
sort_results.to_excel('RQ1_Regression_Ratio.xlsx')


# Actually worse values.  A bit suprised the ensemble methods are not doing better.  Next step will do a logistic/classification.

# In[ ]:


xl_full = df.drop(['delivery_count_sum','is_above_average_delivery_day','log_ratio_to_average'],axis=1)
xl_full = pd.get_dummies(xl_full,drop_first = True)                #make dummies for categorical values for analysis
yl = df['is_above_average_delivery_day']

xl_part = xl_full.drop(['PriorDayPrecipitationAmt','Prior2DayPrecipitationAmt','PriorDaySnowfallAmt','Precip_12AM-6AM_No Precip','Precip_6AM-10AM_No Precip','Precip_6AM-12PM_No Precip','Precip_10AM-2PM_No Precip'],axis=1)
vif_calc(xl_part)

aic_val =np.empty(4)
aic_val[0] = aic_calc(xl_part,yl)
temp_x = xl_part.drop(['PriorDayMaximumTemperatureDiff','PriorDayMinimumTemperatureDiff'],axis=1)
aic_val[1] = aic_calc(temp_x,yl)
temp_x = xl_part.drop(['PriorDayMinimumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
aic_val[2] = aic_calc(temp_x,yl)
temp_x = xl_part.drop(['PriorDayMaximumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
aic_val[3] = aic_calc(temp_x,yl)
print(aic_val)

xl_part = xl_part.drop(['PriorDayMaximumTemperatureDiff','PriorDayAverageDryBulbTemperatureDiff'],axis=1)
vif_calc(xl_part)


# Going to utilize 2 datasets in this case -- removing the multi-collinearity is the biggest thing.  Will do a similar bag of models.

# In[ ]:


def run_logistic(X,Y,graph=False,graph_title='Classification Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'fit_intercept' : [True,False],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }    
    cv = StratifiedKFold(n_splits = 10,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=LogisticRegressionCV(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=50,
        scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    #scale the x predictor values and then run the Bayesian search and capture best parameters
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)       
    search.fit(x_scaled,Y)    
    best_params = search.best_params_
    print(best_params)
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,Y,test_size=.24,random_state=5440)
    model = LogisticRegressionCV(cv=cv,fit_intercept=best_params['fit_intercept']
                                 ,solver=best_params['solver'],scoring='accuracy',n_jobs=-1)
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    test_score = model.score(x_test,y_test)
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test))      
    class_groups = len(model.coef_)    
      
    if graph:
        graph_it(y_test,pred_test,graph_title,RQ=4)
        for cg in range(class_groups):
            graph_feature(X.columns,model.coef_[cg],graph_title + ' ("'+ model.classes_[cg]+ '" class)',tree=False)

    return(test_score,test_auc)

# function for fitting trees of various depths using cross-validation
def run_cross_validation_on_classification_RF(X, Y,graph=False,graph_title='Classification Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'n_estimators': [200, 400, 600, 800, 1000],
        'max_depth': (1, 9),
        'criterion': ['gini', 'entropy'],
        'max_features' : ['sqrt','log2']
    }    
    cv = StratifiedKFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=125,     
        scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    search.fit(X,Y)    
    best_params = search.best_params_
    print(best_params)
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.24,random_state=5440)
    model = RandomForestClassifier(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth']
                                   ,criterion=best_params['criterion'],max_features=best_params['max_features'])
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    test_score = model.score(x_test,y_test)
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test), multi_class='ovr', average='weighted')     
      
    if graph:
        graph_it(y_test,pred_test,graph_title,RQ=1)
        graph_feature(X.columns,model.feature_importances_,graph_title)

    return(test_score,test_auc)

def run_cross_validation_on_classification_Boost(X, Y,scoring='accuracy',graph=False,graph_title='Classification Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'n_estimators': [500, 750, 1000, 1250, 1500],
        'max_depth': (1, 9),
        'criterion': ['friedman_mse', 'squared_error'],
        'max_features' : ['sqrt','log2']
    }    
    cv = StratifiedKFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=GradientBoostingClassifier(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=150,
        scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    search.fit(X,Y)    
    best_params = search.best_params_
    print(best_params)
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.24,random_state=5440)
    model = GradientBoostingClassifier(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth']
                                   ,criterion=best_params['criterion'],max_features=best_params['max_features'])
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    test_score = model.score(x_test,y_test)
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test), multi_class='ovr', average='weighted')     
      
    if graph:
        graph_it(y_test,pred_test,graph_title,RQ=4)
        graph_feature(X.columns,model.feature_importances_,graph_title)

    return(test_score,test_auc)


#RDA is Regularized Discriminant Analysis (similar to how elastic-net works with Lasso and Ridge)
def run_RDA_classification(X,Y,graph=False,graph_title='Classification Graph'):
    #X = predictors, Y = response, other numbers for the range of values
    
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search
    hyper_params = {
        'solver' : ['lsqr','eigen'],
        'shrinkage' : np.arange(0,1.005,.005)
    }

    search = BayesSearchCV(
        estimator=LinearDiscriminantAnalysis(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=10,
        n_iter=200,
        scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    #find the hyperparameters on all the data and capture them for use for training and testing
    search.fit(X,Y)    
    best_params = search.best_params_
    print(best_params)    
    
    #scale the X values for consistency (though may not have much effect for LDA as it would knn, PCA, gradient decent and ridge/Lasso...)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,Y,test_size=.24,random_state=5440)
    model = LinearDiscriminantAnalysis(shrinkage=best_params['shrinkage'],solver=best_params['solver'])   
    #model = LinearDiscriminantAnalysis(shrinkage=.9,solver=best['solver'])  
    model.fit(x_train,y_train)
     
    #find the worth of the model  
    pred_test = cross_val_predict(model,x_test,y_test,cv=5,n_jobs=-1)
    pred_score = cross_val_score(model,x_test,y_test,cv=5,n_jobs=-1)
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test))     
    
    class_groups = len(model.coef_)
    
    if graph:
        graph_it(y_test,pred_test,graph_title,RQ=4)
        for cg in range(class_groups):
            graph_feature(X.columns,model.coef_[cg],graph_title + ' ("'+ model.classes_[cg]+ '" class)',tree=False)
         
    return(pred_score.mean(),test_auc)


# In[ ]:


start = timer()
log_accuracy_full,log_auc_full = run_logistic(xl_full,yl,graph=True,graph_title="Weather Only - Logistic Regression - Full")
end = timer()
print(f'Logistic Model on Full Data Complete in {end-start} seconds')

start = timer()
log_accuracy_part,log_auc_part = run_logistic(xl_part,yl,graph=True,graph_title="Weather Only - Logistic Regression - Partial")
end = timer()
print(f'Logistic Model on Data Subset Complete in {end-start} seconds')

start = timer()
rda_accuracy_full,rda_auc_full = run_RDA_classification(xl_full,yl,graph=True,graph_title="Weather Only - Discriminant Analysis - Full")
end = timer()
print(f'RDA Model on Full Data Complete in {end-start} seconds')

start = timer()
rda_accuracy_part,rda_auc_part = run_RDA_classification(xl_part,yl,graph=True,graph_title="Weather Only - Discriminant Analysis - Partial")
end = timer()
print(f'RDA Model on Data Subset Complete in {end-start} seconds')

start = timer()
rf_accuracy_part,rf_auc_part = run_cross_validation_on_classification_RF(xl_part,yl,graph=True,graph_title="Weather Only - Random Forests - Partial")
end = timer()
print(f'Random Forest Model on Data Subset Complete in {end-start} seconds')

start = timer()
boost_accuracy_full,boo_auc_full = run_cross_validation_on_classification_Boost(xl_full,yl,graph=True,graph_title="Weather Only - Boosted Trees - Full")
end = timer()
print(f'Boosted Trees Model on Full Dataset Complete in {end-start} seconds')

start = timer()
boost_accuracy_part,boo_auc_part = run_cross_validation_on_classification_Boost(xl_part,yl,graph=True,graph_title="Weather Only - Boosted Trees - Partial")
end = timer()
print(f'Boosted Trees Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#create a data frame of the results for analysis
result_aa_list  = [['Logistic Run 1','Full',log_accuracy_full,log_auc_full]
                  ,['Logistic Run 2','Partial',log_accuracy_part,log_auc_part]
                  ,['RDA Run 1','Full',rda_accuracy_full,rda_auc_full]
                  ,['RDA Run 2','Partial',rda_accuracy_part,rda_auc_part]
                  ,['Random Forest Run 1','Partial',rf_accuracy_part,rf_auc_part]
                  ,['Boosted Trees Run 1','Full',boost_accuracy_full,boo_auc_full]
                  ,['Boosted Trees Run 2','Partial',boost_accuracy_part.boo_auc_part]]
results_above_average = pd.DataFrame(result_aa_list,columns=['Model','Dataset','Accuracy','AUC'])
sort_results = results_above_average.sort_values(['AUC','Accuracy'],ascending=[False,False])
sort_results.to_excel('RQ1_Ticket_Classification.xlsx')


# In[ ]:


full_end = timer()
print(f'Total elapsed time {(full_end-full_start)/60}')

