#!/usr/bin/env python
# coding: utf-8

# ## Does what happens in the morning affect when deliveries occur?

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
from skopt.space import Integer, Real, Categorical 
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

def graph_feature(names,fi,graph_title,tree=True,add_label=True):
    sort_key = fi.argsort()
    plt.figure(figsize=(5,8))
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


# In[ ]:


#models used for classification
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
    print(graph_title,file=open('RQ4_hyperparameters','a'))
    print(best_params,file=open('RQ4_hyperparameters','a'))
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(x_scaled,Y,test_size=.24,random_state=5440)
    model = LogisticRegressionCV(cv=cv,fit_intercept=best_params['fit_intercept']
                                 ,solver=best_params['solver'],scoring='accuracy',n_jobs=-1)
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    test_score = model.score(x_test,y_test)
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test), multi_class='ovr', average='weighted')      
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
    print(graph_title,file=open('RQ4_hyperparameters','a'))
    print(best_params,file=open('RQ4_hyperparameters','a'))
    
    #now that the best parameters are found, split the data, run on a test dataset and then predict results
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.24,random_state=5440)
    model = RandomForestClassifier(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth']
                                   ,criterion=best_params['criterion'],max_features=best_params['max_features'])
    model.fit(x_train,y_train)
    pred_test = model.predict(x_test)
    test_score = model.score(x_test,y_test)
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test), multi_class='ovr', average='weighted')     
      
    if graph:
        graph_it(y_test,pred_test,graph_title,RQ=4)
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
    print(graph_title,file=open('RQ4_hyperparameters','a'))
    print(best_params,file=open('RQ4_hyperparameters','a'))
    
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
    print(graph_title,file=open('RQ4_hyperparameters','a'))
    print(best_params,file=open('RQ4_hyperparameters','a'))
    
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
    test_auc = roc_auc_score(y_test,model.predict_proba(x_test), multi_class='ovr', average='weighted')     
    
    class_groups = len(model.coef_)
    
    if graph:
        graph_it(y_test,pred_test,graph_title,RQ=4)
        for cg in range(class_groups):
            graph_feature(X.columns,model.coef_[cg],graph_title + ' ("'+ model.classes_[cg]+ '" class)',tree=False)
         
    return(pred_score.mean(),test_auc)


# In[ ]:


#other functions
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

#function to run baseline regression
def get_stats(x,y,log=False):
    if (log == True):
        results = sm.Logit(y,x).fit()
    else: results = sm.OLS(y,x).fit()
    print(results.summary())


# In[ ]:


#load the soybean dataset into a dataframe and confirm values
full_start = timer()
df_soy_raw = pd.read_csv('DataSets\\Savage_AM_PM_Soybeans_Tickets.csv')
df_soy_raw.head(12)


# In[ ]:


#Load the corn dataset into a dataframe and confirm the values
df_corn_raw = pd.read_csv('DataSets\\Savage_AM_PM_Corn_Tickets.csv')
df_corn_raw.head(12)


# Response variables are PM_Diff_Amt (regression).  A positive number represents more afternoon activity.
# 
# Prep both corn and soybean datasets and create both a "full" and "partial" and one that only has significant p-values

# In[ ]:


df_soy = df_soy_raw.copy()
df_corn = df_corn_raw.copy()

#fill in null values with mean (only 21 out of 40K+)
#df_soy['DailyAverageSeaLevelPressure'] = df_soy['DailyAverageSeaLevelPressure'].fillna(df_soy['DailyAverageSeaLevelPressure'].mean())
#df_corn['DailyAverageSeaLevelPressure'] = df_corn['DailyAverageSeaLevelPressure'].fillna(df_corn['DailyAverageSeaLevelPressure'].mean())

#create full data sets for each type of analysis
ys_class = df_soy['PM_Diff_Category_Val']
xs_full = df_soy.drop(['PM_Diff_Amt','PM_Diff_Category_Val'],axis=1)
xs_full = pd.get_dummies(xs_full,drop_first = True) #make dummies for categorical values

yc_class = df_corn['PM_Diff_Category_Val']
xc_full = df_corn.drop(['PM_Diff_Amt','PM_Diff_Category_Val'],axis=1)
xc_full = pd.get_dummies(xc_full,drop_first = True) #make dummies for categorical values


# In[ ]:


#take a look  at the vif calcuations for soybeans
vif_calc(xs_full)
#open_price and prior_day_open_price looked to be related.  Leave the open_price due to the nature of the study
xs_part = xs_full.drop(['prior_day_open_price'],axis=1)
vif_calc(xs_part)

#remove a "no precip" categorizations and a wind category and it should be good enough
xs_part = xs_part.drop(['Precip_6AM-12PM_No Precip','Precip_12AM-6AM_No Precip','WindSpeed_12AM-6AM_avg_val'],axis=1)
vif_calc(xs_part)


# Lack of multi-collinearity is most likely due to more direct feature selection.  For example, only grabbing the morning pieces.

# In[ ]:


#take a look  at the vif calcuations for corn
vif_calc(xc_full)
#open_price and prior_day_open_price looked to be related.  Leave the open_price due to the nature of the study
xc_part = xc_full.drop(['prior_day_open_price'],axis=1)
vif_calc(xc_part)

#remove a "no precip" categorizations and should be good
xc_part = xc_part.drop(['Precip_6AM-12PM_No Precip','Precip_12AM-6AM_No Precip','WindSpeed_12AM-6AM_avg_val'],axis=1)
vif_calc(xc_part)


# In[ ]:


#use the same "sig" as what was found in the regressions
xs_sig = xs_part.drop(['Precip_6AM-12PM_Precip','Precip_12AM-6AM_Precip','yesterday_avg_trend_Significantly Worse (<-.10)','yesterday_avg_trend_Worse (<-.05)','yesterday_avg_trend_Worse (<-.05)','yesterday_avg_trend_Slightly Worse (<-.02)','prior_day_open_diff','prior_day_trend_Slightly Worse (<-.02)','yesterday_avg_trend_Nominal Change','prior_day_trend_Worse (<-.05)','prior_day_trend_Nominal Change','yesterday_avg_diff','prior_day_trend_Significantly Worse (<-.10)','prior_day_trend_Slightly Better (>.02)','yesterday_avg_trend_Slightly Better (>.02)','WindSpeed_10AM-2PM_avg_val','WindSpeed_6AM-10AM_avg_val','prior_day_trend_Significantly Better (>.10)','open_price'],axis=1)
xc_sig = xc_part.drop(['WindSpeed_10AM-2PM_avg_val','prior_day_trend_Significantly Better (>.10)','yesterday_avg_trend_Significantly Better (>.10)','Precip_6AM-12PM_Precip','prior_day_trend_Significantly Worse (<-.10)','prior_day_open_diff','prior_day_trend_Slightly Better (>.02)'],axis=1)


# In[ ]:


#Soybeans Classfication
start = timer()
log_accuracy_part,log_auc_part = run_logistic(xs_part,ys_class,graph=True,graph_title="Soybeans - Logistic Regression - Partial")
end = timer()
print(f'Logistic Model on Data Subset Complete in {end-start} seconds')

start = timer()
log_accuracy_sig,log_auc_sig = run_logistic(xs_sig,ys_class,graph=True,graph_title="Soybeans - Logistic Regression - Sig P")
end = timer()
print(f'Logistic Model on Sig P Data Complete in {end-start} seconds')

start = timer()
rda_accuracy_full,rda_auc_full = run_RDA_classification(xs_full,ys_class,graph=True,graph_title="Soybeans - Discriminant Analysis - Full")
end = timer()
print(f'RDA Model on Full Data Complete in {end-start} seconds')

start = timer()
rda_accuracy_part,rda_auc_part = run_RDA_classification(xs_part,ys_class,graph=True,graph_title="Soybeans - Discriminant Analysis - Partial")
end = timer()
print(f'RDA Model on Data Subset Complete in {end-start} seconds')

start = timer()
rda_accuracy_sig,rda_auc_sig = run_RDA_classification(xs_sig,ys_class,graph=True,graph_title="Soybeans - Discriminant Analysis - Sig P")
end = timer()
print(f'RDA Model on Sig P Complete in {end-start} seconds')

start = timer()
rf_accuracy_part,rf_auc_part = run_cross_validation_on_classification_RF(xs_part,ys_class,graph=True,graph_title="Soybeans - Random Forests - Partial")
end = timer()
print(f'Random Forest Model on Data Subset Complete in {end-start} seconds')

start = timer()
boost_accuracy_full,boo_auc_full = run_cross_validation_on_classification_Boost(xs_full,ys_class,graph=True,graph_title="Soybeans - Boosted Trees - Full")
end = timer()
print(f'Boosted Trees Model on Full Dataset Complete in {end-start} seconds')

start = timer()
boost_accuracy_part,boo_auc_part = run_cross_validation_on_classification_Boost(xs_part,ys_class,graph=True,graph_title="Soybeans - Boosted Trees - Partial")
end = timer()
print(f'Boosted Trees Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#Soybeans Classification Results
#create a data frame of the results for analysis
result_aa_list  = [['Logistic Run 1','Partial',log_accuracy_part,log_auc_part]
                  ,['Logistic Run 2','SigP',log_accuracy_part,log_auc_sig]                   
                  ,['RDA Run 1','Full',rda_accuracy_full,rda_auc_full]
                  ,['RDA Run 2','Partial',rda_accuracy_part,rda_auc_part]
                  ,['RDA Run 3','SigP',rda_accuracy_sig,rda_auc_sig]                   
                  ,['Random Forest Run 1','Partial',rf_accuracy_part,rf_auc_part]
                  ,['Boosted Trees Run 1','Full',boost_accuracy_full,boo_auc_full]
                  ,['Boosted Trees Run 2','Partial',boost_accuracy_part,boo_auc_part]]
results_above_average = pd.DataFrame(result_aa_list,columns=['Model_Name','Data_Used','Accuracy','AUC'])
sort_results = results_above_average.sort_values(['AUC','Accuracy'],ascending=[False,False])
sort_results.to_excel('RQ4_Soybean_Classification.xlsx')


# In[ ]:


#Corn Classfication
start = timer()
log_accuracy_part,log_auc_part = run_logistic(xc_part,yc_class,graph=True,graph_title="Corn - Logistic Regression - Partial")
end = timer()
print(f'Logistic Model on Data Subset Complete in {end-start} seconds')

start = timer()
log_accuracy_sig,log_auc_sig = run_logistic(xc_sig,yc_class,graph=True,graph_title="Corn - Logistic Regression - Sig P")
end = timer()
print(f'Logistic Model on Sig P Data Complete in {end-start} seconds')

start = timer()
rda_accuracy_full,rda_auc_full = run_RDA_classification(xc_full,yc_class,graph=True,graph_title="Corn - Discriminant Analysis - Full")
end = timer()
print(f'RDA Model on Full Data Complete in {end-start} seconds')

start = timer()
rda_accuracy_part,rda_auc_part = run_RDA_classification(xc_part,yc_class,graph=True,graph_title="Corn - Discriminant Analysis - Partial")
end = timer()
print(f'RDA Model on Data Subset Complete in {end-start} seconds')

start = timer()
rda_accuracy_sig,rda_auc_sig = run_RDA_classification(xc_sig,yc_class,graph=True,graph_title="Corn - Discriminant Analysis - Sig P")
end = timer()
print(f'RDA Model on Sig P Complete in {end-start} seconds')

start = timer()
rf_accuracy_part,rf_auc_part = run_cross_validation_on_classification_RF(xc_part,yc_class,graph=True,graph_title="Corn - Random Forests - Partial")
end = timer()
print(f'Random Forest Model on Data Subset Complete in {end-start} seconds')

start = timer()
boost_accuracy_full,boo_auc_full = run_cross_validation_on_classification_Boost(xc_full,yc_class,graph=True,graph_title="Corn - Boosted Trees - Full")
end = timer()
print(f'Boosted Trees Model on Full Dataset Complete in {end-start} seconds')

start = timer()
boost_accuracy_part,boo_auc_part = run_cross_validation_on_classification_Boost(xc_part,yc_class,graph=True,graph_title="Corn - Boosted Trees - Partial")
end = timer()
print(f'Boosted Trees Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#Corn  Classification Results
#create a data frame of the results for analysis
c_result_aa_list  = [['Logistic Run 1','Partial',log_accuracy_part,log_auc_part]
                  ,['Logistic Run 2','SigP',log_accuracy_part,log_auc_sig]                   
                  ,['RDA Run 1','Full',rda_accuracy_full,rda_auc_full]
                  ,['RDA Run 2','Partial',rda_accuracy_part,rda_auc_part]
                  ,['RDA Run 3','SigP',rda_accuracy_sig,rda_auc_sig]                   
                  ,['Random Forest Run 1','Partial',rf_accuracy_part,rf_auc_part]
                  ,['Boosted Trees Run 1','Full',boost_accuracy_full,boo_auc_full]
                  ,['Boosted Trees Run 2','Partial',boost_accuracy_part,boo_auc_part]]
results_above_average = pd.DataFrame(c_result_aa_list,columns=['Model','Dataset','Accuracy','AUC'])
c_sort_results = results_above_average.sort_values(['AUC','Accuracy'],ascending=[False,False])
c_sort_results.to_excel('RQ4_Corn_Classification.xlsx')


# In[ ]:


full_end = timer()
print(f'time elapsed in minutes = {(full_end-full_start)/60}')

