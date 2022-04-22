#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score,cross_val_predict,RepeatedStratifiedKFold,RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
#need "pip install scikit-optimize"
from skopt.searchcv import BayesSearchCV
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


#functions to perform graphing
def graph_it(y_true,y_pred,title="Graph",RQ=1):
# do the different graphing
    plt.rcParams.update({'font.sans-serif':'Arial'})
    plt.figure(figsize=(10,6))

    if (RQ == 1):
        lables = np.array(False,True)
    else: labels = np.array(['Delivered Early','Within 0-25%','Within 25-50%','Within 50-75%','Within 75-100%','Delivered Late'])
    
    #confusion matrix
    cm = confusion_matrix(y_true,y_pred,labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels= np.array(['Early','0-25%','25-50%','50-75%','75-100%','Late']))
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
        if (full_count > 15):
            label_divisor = 3.5
        else: label_divisor = 2.5
        exit_count=full_count
        for bar in bars:
            if(exit_count > 5):
                exit_count = exit_count -1
                continue
            else:
                width = bar.get_width()
                label_y = bar.get_y() + bar.get_height()/label_divisor
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
# function for fitting trees of various depths using cross-validation
def run_cross_validation_on_classification_RF(X, Y,graph=False,graph_title='Classification Graph'):
    #X = predictors, Y = response, log determines if we are using linear or logistic regression
    #first step is to use a Bayes Search algorithm to find the optimal hyperparameters
    #define hyperparameters to search   
    hyper_params = {
        'n_estimators': [500, 750, 1000, 1250],
        'max_depth': (4, 9),
        'criterion': ['gini', 'entropy'],
        'max_features' : ['sqrt','log2']
    }    
    cv = StratifiedKFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=RandomForestClassifier(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=75,     
        scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    search.fit(X,Y)    
    best_params = search.best_params_
    print(graph_title,file=open('RQ3_hyperparameters_class_only','a'))
    print(best_params,file=open('RQ3_hyperparameters_class_only','a'))
    
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
        'n_estimators': [500, 750, 1000, 1250],
        'max_depth': (4, 9),
        'criterion': ['friedman_mse', 'squared_error'],
        'max_features' : ['sqrt','log2']
    }    
    cv = StratifiedKFold(n_splits = 5,shuffle=True,random_state=5440)   #set random_state to make results repeatable
    search = BayesSearchCV(
        estimator=GradientBoostingClassifier(),
        search_spaces=hyper_params,
        n_jobs=-1,
        cv=cv,
        n_iter=75,
        scoring="accuracy",
        verbose=0,
        random_state=5440
    )
    
    search.fit(X,Y)    
    best_params = search.best_params_
    print(graph_title,file=open('RQ3_hyperparameters_class_only','a'))
    print(best_params,file=open('RQ3_hyperparameters_class_only','a'))
    
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
df_soy_raw = pd.read_csv('DataSets\\Soybean_Contracted_Deliveries_Export.csv')
df_soy_raw.head(12)


# In[ ]:


#Load the corn dataset into a dataframe and confirm the values
df_corn_raw = pd.read_csv('DataSets\\Corn_Contracted_Deliveries_Export.csv')
df_corn_raw.head(12)


# Response variables are Diff_from_average_amt (regression) and Diff_from_average_category (classification)
# 
# Prep both corn and soybean datasets and create both a "full" and "partial"

# In[ ]:


df_soy = df_soy_raw.copy()
df_corn = df_corn_raw.copy()

#drop fields not needed for analysis
df_soy = df_soy.drop(['commodity_name','net_freight_weight_qty','ship_to_range_val','days_from_deadline_val','delivery_date_day'],axis=1)
df_corn = df_corn.drop(['commodity_name','net_freight_weight_qty','ship_to_range_val','days_from_deadline_val','delivery_date_day'],axis=1)

#fill in null values with mean (only 21 out of 40K+)
df_soy['DailyAverageSeaLevelPressure'] = df_soy['DailyAverageSeaLevelPressure'].fillna(df_soy['DailyAverageSeaLevelPressure'].mean())
df_corn['DailyAverageSeaLevelPressure'] = df_corn['DailyAverageSeaLevelPressure'].fillna(df_corn['DailyAverageSeaLevelPressure'].mean())
df_soy['DailyAverageStationPressure'] = df_soy['DailyAverageStationPressure'].fillna(df_soy['DailyAverageStationPressure'].mean())
df_corn['DailyAverageStationPressure'] = df_corn['DailyAverageStationPressure'].fillna(df_corn['DailyAverageStationPressure'].mean())
df_soy['DailyAverageRelativeHumidity'] = df_soy['DailyAverageRelativeHumidity'].fillna(df_soy['DailyAverageRelativeHumidity'].mean())
df_corn['DailyAverageRelativeHumidity'] = df_corn['DailyAverageRelativeHumidity'].fillna(df_corn['DailyAverageRelativeHumidity'].mean())

#remove boolean
df_soy['is_midweek'] = df_soy['is_midweek'].astype(str)
df_corn['is_midweek'] = df_corn['is_midweek'].astype(str)


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


#take a look  at the vif calcuations for soybeans
vif_calc(xs_full)

xs_part = xs_full.drop(['max_price','prior_2_day_open_price','prior_day_open_price','DailyAverageStationPressure','DailyAverageRelativeHumidity','prior_2_day_open_diff','prior_3_day_open_diff'],axis=1)
vif_calc(xs_part)



# Multi-colinearity is present, but not as bad as I've seen.  A bit more suprised at what is showing multicollinearity, since I don't have other price numeric fields not anything with relative humidity.  Do a check on the correlation of the ones that are heavily multi-collinear to see which would make sense to remove.

# In[ ]:


#do aic checks on which windspeed to remove
aic_val =np.empty(4)
aic_val[0] = aic_calc(xs_part,ys_reg)
temp_x = xs_part.drop(['close_price','prior_3_day_open_price'],axis=1)
aic_val[1] = aic_calc(temp_x,ys_reg)
temp_x = xs_part.drop(['close_price','recent_avg_price'],axis=1)
aic_val[2] = aic_calc(temp_x,ys_reg)
temp_x = xs_part.drop(['prior_3_day_open_price','recent_avg_price'],axis=1)
aic_val[3] = aic_calc(temp_x,ys_reg)
print(aic_val)

xs_part = xs_part.drop(['recent_avg_price_diff','prior_day_open_diff','DailyAverageSeaLevelPressure','recent_avg_price','prior_3_day_open_price'],axis=1)
vif_calc(xs_part)


# In[ ]:


#do the same for corn
vif_calc(xc_full)
#remove infinite values

xc_part = xc_full.drop(['max_price','prior_2_day_open_price','prior_day_open_price','DailyAverageStationPressure','DailyAverageRelativeHumidity','prior_2_day_open_diff','prior_3_day_open_diff'],axis=1)
vif_calc(xc_part)

xc_part = xc_part.drop(['recent_avg_price_diff','prior_day_open_diff','DailyAverageSeaLevelPressure','recent_avg_price','prior_3_day_open_price'],axis=1)
vif_calc(xc_part)


# In[ ]:


#Soybeans Classfication
start = timer()
rf_accuracy_part,rf_auc_part = run_cross_validation_on_classification_RF(xs_part,ys_class,graph=True,graph_title="Contract Soybeans - Random Forests - Partial")
end = timer()
print(f'Random Forest Model on Data Subset Complete in {end-start} seconds')

start = timer()
boost_accuracy_full,boost_auc_full = run_cross_validation_on_classification_Boost(xs_full,ys_class,graph=True,graph_title="Contract Soybeans - Boosted Trees - Full")
end = timer()
print(f'Boosted Trees Model on Full Dataset Complete in {end-start} seconds')

start = timer()
boost_accuracy_part,boost_auc_part = run_cross_validation_on_classification_Boost(xs_part,ys_class,graph=True,graph_title="Contract Soybeans - Boosted Trees - Partial")
end = timer()
print(f'Boosted Trees Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#Soybeans Classification Results
#create a data frame of the results for analysis
result_aa_list  = [      
                   ['Random Forest Run 1 -- Class Run','Partial',rf_accuracy_part,rf_auc_part]
                  ,['Boosted Trees Run 1 -- Class Run','Full',boost_accuracy_full,boost_auc_full]    
                  ,['Boosted Trees Run 2 -- Class Run','Partial',boost_accuracy_part,boost_auc_part]]
results_above_average = pd.DataFrame(result_aa_list,columns=['Model','Dataset','Accuracy','AUC'])
sort_results = results_above_average.sort_values(['AUC','Accuracy'],ascending=[False,False])
sort_results.to_excel('RQ3_Soybean_Classification_new_run.xlsx')


# In[ ]:


#Corn Classfication
start = timer()
rf_accuracy_part,rf_auc_part = run_cross_validation_on_classification_RF(xc_part,yc_class,graph=True,graph_title="Contract Corn - Random Forests - Partial")
end = timer()
print(f'Random Forest Model on Data Subset Complete in {end-start} seconds')

start = timer()
boost_accuracy_full,boost_auc_full = run_cross_validation_on_classification_Boost(xc_full,yc_class,graph=True,graph_title="Contract Corn - Boosted Trees - Full")
end = timer()
print(f'Boosted Trees Model on Full Dataset Complete in {end-start} seconds')

start = timer()
boost_accuracy_part,boost_auc_part = run_cross_validation_on_classification_Boost(xc_part,yc_class,graph=True,graph_title="Contract Corn - Boosted Trees - Partial")
end = timer()
print(f'Boosted Trees Model on Data Subset Complete in {end-start} seconds')


# In[ ]:


#corn Classification Results
#create a data frame of the results for analysis
c_result_aa_list  = [            
                   ['Random Forest Run 1 -- Class Run','Partial',rf_accuracy_part,rf_auc_part]
                  ,['Boosted Trees Run 1 -- Class Run','Full',boost_accuracy_full,boost_auc_full]    
                  ,['Boosted Trees Run 2 -- Class Run','Partial',boost_accuracy_part,boost_auc_part]]
results_above_average = pd.DataFrame(c_result_aa_list,columns=['Model','Dataset','Accuracy','AUC'])
c_sort_results = results_above_average.sort_values(['AUC','Accuracy'],ascending=[False,False])
c_sort_results.to_excel('RQ3_Corn_Classification_new_run.xlsx')

