#Lucas Bouchard



import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
#import matplotlib.pyplot as plt  
#%matplotlib inline

#https://pythonspot.com/linear-regression/

# Using the candy-data.csv file in the repo, populate an AnalysisData object that will hold the data you'll use for today's problem set. You should read in the data from the CSV, store the data in the dataset variable, and initialize the xs (column name) and targetY variables appropriately. targetY should reference the variable describing whether or not a candy is chocolate.

#AnalysisData
#AnalysisData, which will have, at a minimum, attributes called dataset (which holds the parsed dataset) and variables (which will hold a list containing the indexes for all of the variables in your data). 
class AnalysisData:

#Initialize attributes
    def __init__(self):
        self.dataset=[]
        self.X_variables=[]
    
        
    
#function that opens file and removes string columns
    def parserFile(self, filename):
        self.dataset=pd.read_csv(filename)

        #Exclude uncomparable variables from X_variables
        #self.X_variables=self.dataset[[:,1:12]]/1-9
        for variable in self.dataset.columns.values:
            if variable != "competitorname":
                self.X_variables.append(variable)
                
candy_data = AnalysisData()
candy_data.parserFile('candy-data.csv')
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html          
#LinearAnalysis
#Which will contain your functions for doing linear regression and have at a minimum attributes called bestX (which holds the best X predictor for your data), targetY (which holds the index to the target dependent variable), and fit (which will hold how well bestX predicts your target variable).
class LinearAnalysis:
    
    def __init__(self,target_Y):
        self.bestX =None
        self.targetY = target_Y
        self.fit=None
    
    def runSimpleAnalysis(self, data):
        linear_r2=-1
        best_linear_variable=None
        #establish independent variable
        for column in data.X_variables:
            if column != self.targetY:
                Y_variable= data.dataset[column].values
                Y_variable=Y_variable.reshape(len(Y_variable),1)
                #Regression 
                regression = LinearRegression()
                regression.fit(Y_variable, data.dataset[self.targetY])
                r_score = regression.predict(Y_variable)
                r_score = r2_score(data.dataset[self.targetY],Y_variable)
                if r_score > linear_r2:
                    linear_r2 = r_score
                    best_linear_variable = column
        self.bestX = best_linear_variable
        print("Best lin predictor is " + self.bestX + " at ", linear_r2)
        #print(best_linear_variable, linear_r2)
        print('Linear Regression Analysis coefficients: ', regression.coef_)
        print('Linear Regression Analysis intercept: ', regression.intercept_)
        
        
    
class LogisticAnalysis:
    
    def __init__(self, target_Y):
        self.bestX = None
        self.targetY = target_Y
        self.fit = None
        self.type= int
    def runSimpleAnalysis2(self, data):
        r2=-1
        best_variable=data.dataset
        #establish independent variable
        for column in data.X_variables:
            if column != self.targetY:
                Y_variable= data.dataset[column].values
                Y_variable=Y_variable.reshape(len(Y_variable),1)
                #Regression 
                regression = LogisticRegression(solver='lbfgs')
                regression.fit(Y_variable, data.dataset[self.targetY])
                r_score = regression.predict(Y_variable)
                r_score = r2_score(data.dataset[self.targetY],r_score)
                if r_score > r2:
                    r2 = r_score
                    best_variable = column
        self.bestX = best_variable
        print("Best log predictor is " + self.bestX + " at ", r2)
        print('Logistic Regression Analysis Coefficients: ', regression.coef_)
        print('Logistic Regression Analysis Intercept: ', regression.intercept_)

   
        
        
    #def runMultipleRegression(self, data):
        #r2=-1
        #best_variable=data.dataset

#PROBLEM 1
#Add a function to the LogisticAnalysis object called runSimpleAnalysis. This function should take in an AnalysisData object as a parameter and should use this object to compute which variable best predicts whether or not a candy is chocolate using logistic regression. Print the variable name and the resulting fit. Do the two functions find the same optimal variable? Which method best fits this data? Make sure your best predictor is NOT the same as the targetY variable.








#PROBLEM 2. Create a function to initialize a LinearAnalysis object that takes a targetY as its input parameter. Create the same function for LogisticAnalysis. Note that you will use the LinearAnalysis object to try to predict the amount of sugar in the candy and the LogisticAnalysis object to predict whether or not the candy is chocolate.
#ABOVE

#Problem 3
candy_data_lin_analysis = LinearAnalysis('chocolate')
candy_data_lin_analysis.runSimpleAnalysis(candy_data)
candy_data_log_analysis = LogisticAnalysis('chocolate')
candy_data_log_analysis.runSimpleAnalysis2(candy_data)



#Problem 4
#a.) independent=All the different types of candies(Categorical Variable)
#dependent= sugar percent(continous variable)
#null hypothesis: 

     
        
        
        
        
        
        

    

