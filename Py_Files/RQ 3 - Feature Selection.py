#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from skopt.space import Integer, Real, Categorical 
from skopt.utils import use_named_args
from skopt import gp_minimize
from timeit import default_timer as timer
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#functions to perform graphing
def graph_it(y_true,y_pred,title="Graph",RQ=1):
# do the different graphing
    plt.rcParams.update({'font.sans-serif':'Arial'})

    if (RQ == 1):
        lables = np.array(False,True)
    else: labels = np.array(['Delivered Early','Within 0-25%','Within 25-50%','Within 50-75%','Within 75-100%','Delivered Late'])
    
    #confusion matrix
    cm = confusion_matrix(y_true,y_pred,labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot(cmap='Greys',colorbar=False)
    plt.title(title)            
    plt.show()

def graph_feature(names,fi,graph_title,tree=True,add_label=True):
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
                label_y = bar.get_y() + bar.get_height() /4
                plt.text(.01, label_y, s=f'{width:.4f}',fontweight='bold',color='black')
                exit_count = exit_count - 1
    elif (add_label==True and tree==False):
        # Add annotation to top and bottom 3 bars
        plt.xlabel('Coefficients') 
        full_count=len(bars)
        exit_count=full_count
        for bar in bars:
            if(exit_count > 3 and exit_count <= full_count-3 ):
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


# In[3]:


#enet regression:  handles E-Net and Lasso
def run_enet_regression(X,Y,graph=False,graph_title='Regression Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'alpha' : [.0001,.001,.01,.1,1.0,10,100],
        'l1_ratio' : [.01,.05,.2,.5,.8,.95,.99,1],
        'fit_intercept' : [True,False]
    }    
    cv = KFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=ElasticNet(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=50,
        #scoring="accuracy",  -- leave as default which is based on the estimator
        verbose=0,
        random_state=5440
    )
    #scale the x predictor values and then run the Bayesian search and capture best parameters
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)       
    search.fit(x_scaled,Y)    
    best_params = search.best_params_
    print(graph_title,file=open('RQ3_hyperparameters','a'))
    print(best_params,file=open('RQ3_hyperparameters','a'))
    
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
   
    cv = RepeatedKFold(n_splits = 5,n_repeats=25,random_state=5440)  
    model = RidgeCV(alphas=[.0001,.001,.01,.1,1.0,10,100],cv=cv)
    model.fit(x_scaled,Y)
    print(graph_title,file=open('RQ3_hyperparameters','a'))
    print(model.alpha_,file=open('RQ3_hyperparameters','a'))

    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,Y,test_size=.24,random_state=5440)   

    pred_test = model.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_test,pred_test))
    r2_test = r2_score(y_test, pred_test)  
    
    if graph:
        graph_feature(X.columns,model.coef_,graph_title,tree=False)
    
    return(rmse_test,r2_test)


# In[4]:


# function for fitting trees of various depths using cross-validation
def run_cross_validation_on_classification_RF(X, Y,graph=False,graph_title='Classification Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'n_estimators': [500],
        'max_depth': (1, 9),
        'criterion': ['gini'],
        'max_features' : ['sqrt','log2']
    }    
    cv = StratifiedKFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=25,     
        scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    search.fit(X,Y)    
    best_params = search.best_params_
    print(graph_title,file=open('RQ3_hyperparameters','a'))
    print(best_params,file=open('RQ3_hyperparameters','a'))
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.24,random_state=5440)
    model = RandomForestClassifier(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth']
                                   ,criterion=best_params['criterion'],max_features=best_params['max_features'])
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    test_score = model.score(x_test,y_test)
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test), multi_class='ovr', average='weighted')     
      
    if graph:
        graph_it(y_test,pred_test,graph_title,RQ=3)
        graph_feature(X.columns,model.feature_importances_,graph_title)

    return(test_score,test_auc)

def run_cross_validation_on_classification_Boost(X, Y,scoring='accuracy',graph=False,graph_title='Classification Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'n_estimators': [500],
        'max_depth': (1, 9),
        'criterion': ['squared_error'],
        'max_features' : ['sqrt','log2']
    }    
    cv = StratifiedKFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=GradientBoostingClassifier(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=25,
        scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    search.fit(X,Y)    
    best_params = search.best_params_
    print(graph_title,file=open('RQ3_hyperparameters','a'))
    print(best_params,file=open('RQ3_hyperparameters','a'))
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.24,random_state=5440)
    model = GradientBoostingClassifier(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth']
                                   ,criterion=best_params['criterion'],max_features=best_params['max_features'])
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    test_score = model.score(x_test,y_test)
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test), multi_class='ovr', average='weighted')     
      
    if graph:
        graph_it(y_test,pred_test,graph_title,RQ=3)
        graph_feature(X.columns,model.feature_importances_,graph_title)

    return(test_score,test_auc)


# In[6]:


#load the soybean dataset into a dataframe and confirm values
full_start = timer()
df_soy_raw = pd.read_csv('DataSets\\Soybean_Contracted_Deliveries_Export.csv')
df_soy_raw.head(12)


# In[7]:


#Load the corn dataset into a dataframe and confirm the values
df_corn_raw = pd.read_csv('DataSets\\Corn_Contracted_Deliveries_Export.csv')
df_corn_raw.head(12)


# Response variables are Diff_from_average_amt (regression) and Diff_from_average_category (classification)
# 
# Prep both corn and soybean datasets and create both a "full" and "partial"

# In[8]:


df_soy = df_soy_raw.copy()
df_corn = df_corn_raw.copy()

#drop fields not needed for analysis
df_soy = df_soy.drop(['commodity_name','delivery_date_year','net_freight_weight_qty','ship_to_range_val','days_from_deadline_val'],axis=1)
df_corn = df_corn.drop(['commodity_name','delivery_date_year','net_freight_weight_qty','ship_to_range_val','days_from_deadline_val'],axis=1)

#fill in null values with mean (only 21 out of 40K+)
df_soy['DailyAverageSeaLevelPressure'] = df_soy['DailyAverageSeaLevelPressure'].fillna(df_soy['DailyAverageSeaLevelPressure'].mean())
df_corn['DailyAverageSeaLevelPressure'] = df_corn['DailyAverageSeaLevelPressure'].fillna(df_corn['DailyAverageSeaLevelPressure'].mean())

#create full data sets for each type of analysis
ys_reg = df_soy['pct_elapsed_val']
ys_class = df_soy['category_elapsed_val']
xs_full = df_soy.drop(['pct_elapsed_val','category_elapsed_val'],axis=1)
xs_full = pd.get_dummies(xs_full,drop_first = True) #make dummies for categorical values

yc_reg = df_corn['pct_elapsed_val']
yc_class = df_corn['category_elapsed_val']
xc_full = df_corn.drop(['pct_elapsed_val','category_elapsed_val'],axis=1)
xc_full = pd.get_dummies(xc_full,drop_first = True) #make dummies for categorical values


# In[ ]:


start = timer()      
enet_full_rmse,enet_full_r2 = run_enet_regression(xs_full,ys_reg,graph=True,graph_title="Contract Soybeans - Lasso/ENet Regression - Full")
end = timer()
print(f'Enet Regression Model on Full Dataset Complete in {end-start} seconds')


# In[ ]:


start = timer()
rf_accuracy_part,rf_auc_part = run_cross_validation_on_classification_RF(xs_full,ys_class,graph=True,graph_title="Contract Soybeans - Random Forests - Full")
end = timer()
print(f'Random Forest Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#Corn Regression
start = timer()  
ridge_part_rmse,ridge_part_r2 = run_ridge_regression(xc_full,yc_reg,graph=True,graph_title="Contract Corn - Ridge Regression - Full")
end = timer()
print(f'Ridge Regression Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#Corn Classfication
start = timer()
boost_accuracy_full,boost_auc_full = run_cross_validation_on_classification_Boost(xc_full,yc_class,graph=True,graph_title="Contract Corn - Boosted Trees - Full")
end = timer()
print(f'Boosted Trees Model on Full Dataset Complete in {end-start} seconds')

