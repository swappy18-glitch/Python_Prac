#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels as sm
import numpy as np
import scipy
import scipy.stats as stats
import pylab
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy.stats import zscore
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings( "ignore" )
from scipy.stats import kurtosis
from scipy.stats import skew
from statsmodels.stats import weightstats as stests
from scipy.stats import mannwhitneyu
from math import sqrt
from scipy.stats import norm
from scipy.stats import f_oneway
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
pd.options.display.max_rows = 500
pd.options.display.max_columns = 15


# In[2]:


df=pd.read_excel(r'cars.xlsx')
df


# In[3]:


df.dtypes


# In[4]:


df.describe()


# In[5]:


sn.histplot(data=df, x="MPG_City", hue="Type",multiple="stack",binwidth=5)


# In[6]:


sn.histplot(data=df, x="MPG_Highway", hue="Type",multiple="stack",binwidth=5)


# In[7]:


sn.boxplot(data=df, x="MPG_City", y="Type")


# In[8]:


'''Horse power 100 - 250 are found in engine size 4
horse power between 150 to 300 in engine size 6
Horse power between 210 to 250 in engine size 8. Ouliers are found in engine size 8,10 and 12 for the 
Horsepower more than 350'''
sn.scatterplot(data=df, x="Horsepower", y="Cylinders", hue="Cylinders")


# In[9]:


'''Seems Engine size has more importance for the horsepower than Cylinder size, because there is strong 
correlation between the variables. Has the HP increases, EZ also increases.In addition,
data points are close in relation'''
sn.scatterplot(data=df, x="Horsepower", y="EngineSize", hue="EngineSize")


# In[10]:


#groupby the data by delivery type
data = df.groupby("Type")["Invoice"].sum()
data


# In[11]:


data.plot.pie(autopct="%.1f%%")


# In[12]:


##engine size, cylinder, weight, wheelbase, length, and MPG?
df_relation = df[["EngineSize","Cylinders","Weight","Wheelbase","Length","MPG_City","MPG_Highway","MSRP"]]
corr_matrix = df_relation.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
'''MSRP has resonable relation with enginesesize and cylinders. Let`s explore more'''


# In[13]:


'''fillin na in cylinders with mode'''
df_relation['Cylinders'].fillna(df_relation['Cylinders'].mode()[0], inplace=True)
df_relation.Cylinders.isna().sum()


# In[14]:


##How would horse power and gasoline efficiency affect the price
df["MPG_Mean"] = df["MPG_City"]+df["MPG_Highway"]/len(df["MPG_City"]+df["MPG_Highway"])
df["MPG_Mean"].apply(lambda x: round(x,2))
df["MPG_Mean"]


# In[15]:


sn.scatterplot(data=df, x="Horsepower", y="MPG_Mean", hue="Invoice")


# In[16]:


df.groupby("Make").get_group("Porsche")


# In[17]:


df.hist(figsize=(10,10),bins=10)


# In[18]:


def normality(data,feature):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sn.kdeplot(data[feature])
    plt.subplot(1,2,2)
    stats.probplot(data[feature],plot=pylab)
    plt.show()


# In[19]:


df_relation['price_log']=np.log(df_relation['MSRP'])


# In[20]:


#plotting to check the transformation
normality(df_relation,'price_log')


# Let's explore dependent variables more

# In[21]:


class stats2:
    def __init__(self,dataframe):
        self.df = dataframe
        
    def auto_determine(self,col_name):
        return str(self.df[col_name].dtype)

    def pred_outcomes_dtypes(self):
        columns = self.df.columns
        pred = {}
        outcome = {}
        for col in columns:
            type_col= self.auto_determine(col)
            if col[0].lower() == 'x' and col[1]=='_':
                pred[col]= type_col
            elif col[0].lower() == 'y' and col[1]=='_':
                outcome[col]= type_col
        
        return pred,outcome
   
    def our_pred_dtypes(self,outcome): # when x and y dataset not available
        columns = self.df.columns
        pred = {}
        outcomes = {}
        for col in columns:
            
            type_col= self.auto_determine(col)
            if col != outcome:
                pred[col]= type_col
            else:
                outcomes[col]= type_col
                
        return pred,outcomes
                       
    def shapiro_wilk_test(self,predictor,alpha = 0.05):# output is irrelevant here
    # test the null hypothesis for columns given in `cols` of the dataframe `df` under significance level `alpha`.
        fun_stats ={"Predictor" :predictor,"Testname":"Shapiro wilk test","Performance stat Raw":None,"comment":" "}
        col = predictor
        _,p = stats.shapiro(self.df[col])
        fun_stats["Performance stat Raw"]= p
        if p <= alpha:
            fun_stats["comment"] = "The variable is not normally distributed"
        else:
            fun_stats["comment"] ="The variable is normally distributed"
        
        return fun_stats
    
    def kur(self,predictor,alpha= 3): 
        fun_stats={"Predictor":predictor,"Testname":"Kurtosis","Performance stat Raw":None,"comment": " "}
        col = predictor
        value = kurtosis(self.df[col].astype(int), fisher=False)
        fun_stats["Performance stat Raw"]= value
        if value <= alpha:
            fun_stats["comment"]= "The variable",predictor,"is platykurtic, since the value is less than 3"
        elif value == alpha:
            fun_stats["comment"]="The variable",predictor,"is Mesokurtic, since it is at 3 as per pearson`s"
        elif value >= alpha:
            fun_stats["comment"]= "The variable",predictor,"is leptokurtic, since it is beyond 3"
            
        return fun_stats
            
    def skew(self,predictor,alpha = (-0.5,0.5)): #output variable is irrelevant here
        fun_stats={"Predictor":predictor,"Testname":"skewness","Performance stat Raw":None,"comment":" "}
        col = predictor
        value = skew(self.df[col], bias=False)
        fun_stats["Performance stat Raw"]= value
        if value <= alpha[0]:
            fun_stats["comment"]= "The variable is negatively skewed, that is high data concentration on the right side of the graph"
        elif alpha[0] < value < alpha[1]:
            fun_stats["comment"]= "The variable is fairly symmetrical"
        elif value >= alpha[1]:
            fun_stats["comment"]= "The variable is positively skewed, that is high data concentration on the left side of the graph"
        return fun_stats
            
             
    def man_whit(self,predictor,outcome,alpha = 0.5): 
        fun_stats = {"Predictor":predictor,"outcome":outcome,"Testname":"skewness","Performance stat Raw":None,"keep":False,"comment":" "}
        col = predictor
        _,p = mannwhitneyu(self.df[predictor],self.df[outcome])
        fun_stats["Performance stat Raw"]= p
        if p <= alpha:
            fun_stats["keep"] = True
            fun_stats["comment"] = "The variables are correlated"             
        else:
            fun_stats["comment"] = "The variables are not correlated"
            
        return fun_stats
    
  
    def an_ova(self,predictor1,predictor2,predictor3,outcome, alpha = 0.05):
        fun_stats = {'outcome':outcome,"Predictor":[predictor1,predictor2,predictor3],"Testname":"ANOVA","Performance stat Raw":None,"keep":False,"comment":" "}
        col = [predictor1,predictor2,predictor3]
        _,p = f_oneway(self.df[predictor1],self.df[predictor2],self.df[predictor3],self.df[outcome])
        fun_stats["Performance stat Raw"]= p
        if p <= alpha:
            fun_stats["Keep"]=True
            fun_stats["comment"]='No correlation between the variables,since pvalue is:' + str(col) + "which is less than: " + str(alpha)
        else:
            fun_stats["comment"]="The varriables are correlated, since the pvalue is:" + str(col) + "which is more than or equalt to" + str(alpha)
        return fun_stats
        
    
    def chi_sq_tst(self,predictor,outcome,alpha = 0.5):
        fun_stats = {"Predictor":predictor,'outcome':outcome,"Testname":"CHI-SQ","Performance stat Raw":None,"keep":False,"comment":" "}
        data_crosstab = pd.crosstab(self.df[predictor],
                            self.df[outcome],
                           margins=True, margins_name="Total")
        _,p,_,_= chi2_contingency(data_crosstab)
        fun_stats["Performance stat Raw"]= p
        if p<=alpha:
            fun_stats["keep"]=True
            fun_stats["comment"]="The variables are not correlated, since the pvalue is: " + str(p) + "which is  less than:" + str(alpha) 
        else:
            fun_stats["comment"]='The variables are correlated, since pvalue is:' + str(p) + "which is more than:" + str(alpha)
        return fun_stats
    
    def pearson(self,predictor,outcome,alpha = 0.6):
        fun_stats = {"Predictor":predictor,'outcome':outcome,"Testname":"Pearson","Performance stat Raw":None,"keep":False,"comment":" "}
        _,p = pearsonr(self.df[predictor], self.df[outcome])
        fun_stats["Performance stat Raw"]= p
        if p>=alpha:
            fun_stats["keep"]=True
            fun_stats["comment"]="The variables are correlated, since pvalue is:" + str(p) + "which is equal to or more than: " + str(alpha)
        else:
            fun_stats["comment"]='The variables are not correlated, since pvalue is: ' + str(p) +  " which is less than: " + str(alpha)
            
        return fun_stats
        
    
    def ken_dall(self,predictor,outcome,alpha = 0.5):
        fun_stats = {"Predictor":predictor,'outcome':outcome,"Testname":"Kendal","Performance stat Raw":None,"keep":False,"comment":" "}
        _,p = kendalltau(self.df[predictor], self.df[outcome])
        fun_stats["Performance stat Raw"]= p
        if p<=alpha:
            fun_stats["keep"]=True
            fun_stats["comment"]="The variables are correlated, since pvalue is" + str(p) + " which is less than " + str(alpha)
        else:
            fun_stats["comment"]='The variables are not correlated, since pvalue is:'+ str(p) + " which is more than " + str(alpha)
            
        return fun_stats
    
    def spear_man(self,predictor,outcome,alpha = 0.5):
        fun_stats = {"Predictor":predictor,'outcome':outcome,"Testname":"Kendal","Performance stat Raw":None,"keep":False,"comment":" "}
        _,p = spearmanr(self.df[predictor], self.df[outcome])
        fun_stats["Performance stat Raw"]= p
        if p<=alpha:
            
            fun_stats["keep"]=True
            fun_stats["comment"]= 'the variables are correlated, since pvalue is:' + str(p) + 'which is equal to or less than: ' + str(alpha)
        else:
            fun_stats["comment"]='the variables are not correlated, since pvalue is:'+ str(p) + 'which is more than:' + str(alpha)
            
        return fun_stats
    
    def mutual_info(self,predictor,outcome,alpha = 0.5):
        x = np.array(self.df[predictor].astype("category").cat.codes).reshape(-1,1)
        y = self.df[outcome]
        mi_scores = mutual_info_classif(x,y)
        fun_stats = {"Predictor":predictor,'outcome':outcome,"Testname":"Kendal","Performance stat Raw":None,"keep":False,"comment":" "}
        fun_stats["Performance stat Raw"]= mi_scores[0] 
        if mi_scores >=alpha:
            
            fun_stats["keep"]=True
            fun_stats["comment"]= 'the variables are correlated, since pvalue is:' + str(mi_scores) + 'which is equal to or more than: ' + str(alpha)
        else:
            fun_stats["comment"]='the variables are not correlated, since pvalue is:'+ str(mi_scores) + 'which is less than:' + str(alpha)
            
        return fun_stats                          
          
    def main_brain_sys(self,dataframe,predictors,outcomes):
        def check_number(type_arg):
            return ("float" in type_arg or "int" in type_arg)
        stats_result = {"pearson":[],"spearman":[],"chisqtest":[],"mutualinfo":[],"kendall":[]}
        for outcome in outcomes:
            tempOutType=outcomes[outcome]
            
            for predictor in predictors:
                tempPredType = predictors[predictor]
                if check_number(tempOutType) and check_number(tempPredType):
                    temp_res = self.pearson(predictor, outcome)
                    stats_result["pearson"].append(temp_res)
                    temp_res = self.spear_man(predictor,outcome)
                    stats_result["spearman"].append(temp_res)
                elif tempPredType == "object" and tempOutType == "object":
                    temp_res = self.chi_sq_tst(predictor, outcome)
                    stats_result["chisqtest"].append(temp_res)
                    temp_res = self.mutual_info(predictor,outcome)
                    stats_result["mutualinfo"].append(temp_res)
                else:
                    temp_res = self.ken_dall(predictor, outcome)
                    stats_result["kendall"].append(temp_res)
                    
        return stats_result


# In[22]:


y = stats2(df_relation) 
y.shapiro_wilk_test("price_log")


# In[23]:


y.kur("price_log")


# In[24]:


y.skew("price_log")


# In[25]:


'''since the cylinders are ranked and price are continous and both being int, going for kendall,
no linear correlation between dependent and non dependent'''

y.ken_dall("Cylinders","price_log")


# In[26]:


y.shapiro_wilk_test("EngineSize")


# In[27]:


'''not normally distributed two numerical'''
y.spear_man("EngineSize", "price_log")


# In[28]:


'''cate-numerical'''
y.ken_dall("EngineSize", "Cylinders")


# In[30]:


## select the dependent and response variables
df_relation_feature = df_relation[["price_log","EngineSize","Cylinders"]]
df_relation_feature.isna().sum()


# In[31]:


# input 
x = df_relation_feature.iloc[:, 0:2].values


# In[32]:


# # output 
y = df_relation_feature.iloc[:,2].values
np.any(np.isnan(y))


# In[33]:


# input 
X = df_relation_feature.iloc[:, 0:2].values

# # output 
y = df_relation_feature.iloc[:, 2].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 


# Logistic regression with multinomial classification

# In[38]:


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0,multi_class='multinomial') 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test) 


# In[39]:


#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)


# In[45]:


#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred))


# In[47]:


'''Try SVM'''
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(kernel='rbf', C=1).fit(X_train, y_train)
y_pred = svc.predict(X_test)

#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred))


# In[43]:


df_relation.Cylinders.value_counts()


# In[ ]:


'''Precision — What percent of your predictions were correct?
Precision:- Accuracy of positive predictions.
Precision = TP/(TP + FP)
Answer is the class of 4,6,and 8, respectively did better, obviously due to high data'''


# In[ ]:


'''Recall — What percent of the positive cases did you catch?
Recall is the ability of a classifier to find all positive instances.
Recall = TP/(TP+FN)
Again, class 4,6,8 does better'''


# In[ ]:


'''F1 score — What percent of positive predictions were correct?
The F1 score is a weighted harmonic mean of precision and recall 
such that the best score is 1.0 and the worst is 0.0. 
F1 scores are lower than accuracy measures as they embed precision 
and recall into their computation. As a rule of thumb, 
the weighted average of F1 should be used to compare classifier models, not global accuracy.
F1 Score = 2*(Recall * Precision) / (Recall + Precision).
Again class 4,6, and 8 F1 scores are higher and give clarity'''


# In[ ]:


'''Support
Support is the number of actual occurrences of the class in the specified dataset.
That is to say, why class 4,6 and 8s impact was so obvious '''


# In[ ]:


'''compared to svm, logistic regression does a better job'''

