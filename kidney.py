import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml
plt.style.use('seaborn')
import scipy.stats as stats
from scipy.stats import kendalltau
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from scipy.stats import kurtosis
from scipy.stats import skew
from statsmodels.stats import weightstats as stests
from scipy.stats import mannwhitneyu
from math import sqrt
from scipy.stats import norm
from scipy.stats import f_oneway
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr


# In[2]:


data = pd.read_csv("kidney_disease.csv")
df = data.drop("id", axis = 1)
df


# In[3]:


df.info()


# In[4]:


'''Total NAs'''
df_na_count = df.isna().sum().sum()
df_na_count


# In[5]:


'''Total count of values in DF'''
df_total_val = df.count().sum()
df_total_val


# In[6]:


'''Percentage of NAs in entire df'''
na_perc = df_na_count / df_total_val * 100
na_perc
'''around 11% of the data is NA`s, hence going to impute them'''


# In[7]:


df_rep = df[["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo"]]
df_rep.hist()


# In[8]:


df_rep.boxplot(column=["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo"] )


# Since there are presence of outliers and the distribution of the variables are skewed,
# I am replacing NAs with median. Because mean is sensisitve to outliers.

# In[9]:


'''replaced NANs with mode'''
for i in df:
    df[i].fillna(df[i].mode()[0],inplace = True)
    print(i)



# In[10]:


df.isna().sum()


# In[11]:


"remove outliers"
def cap_data(df):
    for col in df.columns:
        print("capping the ",col)
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles = df[col].quantile([0.01,0.99]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col]=df[col]
    return df

final_df=cap_data(df)
final_df


# In[12]:


df_rep_2 = df[["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo"]]
df_rep_2.hist()


# In[13]:


'''qqplot'''
fig, axs = plt.subplots(4,4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
for ax, d in zip(axs.ravel(), df_rep_2):

       stats.probplot(df[d], plot=ax)

       #ax.set_titl(str(d))

plt.show()


# In[14]:


##engine size, cylinder, weight, wheelbase, length, and MPG?
df_cor = df
corr_matrix = df_cor.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()


# In[20]:


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


# In[21]:


for_stats = stats2(df)
predict,outcome = for_stats.our_pred_dtypes("classification")
for_stats.main_brain_sys(df,predict,outcome)

