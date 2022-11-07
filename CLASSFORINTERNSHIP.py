import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import f_oneway
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
from sklearn.feature_selection import mutual_info_classifi
from scipy.stats import pearsonr

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

df = pd.read_csv("kidney_disease.csv").dropna()
df



'''
Testing Functions in Class
FROM SHAPIRO_WILK_TEST FUCNTION to MUTUAL_INFO FUNCTION, focus here is only about the functioning of functions.
Hence, I have considered only "bp","age" irrespective of their respective distribution.
However, the last function which brings in all of the functions together is about the variables(X) type and
the appropriate tests in accordance with the type of Y(here in "classification" is Y)
'''

y = stats2(df)
y.shapiro_wilk_test("bp")

y.kur("bp")

y.skew("bp")

y.man_whit("age","bp")

y.an_ova("age","bp","al","wc")

y.chi_sq_tst("age","bp")

y.pearson("age","bp")

y.spear_man("age","bp")

y.ken_dall("age","bp")

y.spear_man("age","bp")

y.mutual_info("age","bp")

'''Bringing it all together'''
predict,outcome = y.our_pred_dtypes("classification")# when given x and y dataset, the f.our_pred_dtypes function is
#replaced by pred_outcomes_dtypes. predict and outcome are the dictionaries.
y.main_brain_sys(df,predict,outcome)
