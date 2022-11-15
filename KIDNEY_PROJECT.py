
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import where, mean
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
# import SVC classifier
from sklearn.svm import SVC

# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


'''reading kidney file'''
data = pd.read_csv("kidney_disease.csv")
df = data.drop("id", axis = 1)
df

df.info()

'''replacing erroneous ckd value in the classification column'''
df['classification'] = df['classification'].replace(['ckd\t'], 'ckd')

df.classification.value_counts()
'''Total NAs'''
df_na_count = df.isna().sum().sum()
df_na_count


'''Total count of values in DF'''
df_total_val = df.count().sum()
df_total_val

'''Percentage of NAs in entire df'''
na_perc = df_na_count / df_total_val * 100
na_perc
'''around 11% of the data is NA`s, hence going to impute them'''

'''histogram time to see the distribution'''
df_rep = df[["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo"]]
df_rep.hist()


'''boxplot time to see the outliers and data concentration'''
df_rep.boxplot(column=["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo"] )


'''replaced NANs with mode'''
for i in df:
    df[i].fillna(df[i].mode()[0],inplace = True)
    print(i)

'''checking if there is anymore NAs'''
df.isna().sum()

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

'''re-checking the distribution after outliers removal'''
df_rep_2 = df[["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo"]]
df_rep_2.hist()


'''QQplot- TO SEE THE VARIABLE DISTRIBUTION'''
fig, axs = plt.subplots(4,4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
for ax, d in zip(axs.ravel(), df_rep_2):

       stats.probplot(df[d], plot=ax)

plt.show()

'''HEATMAP AND CORRELATION OF NUMERICAL VARIABLES'''
df_cor = df
corr_matrix = df_cor.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

'''THIS IS A STATS CLASS WHICH HAS FUCNTION FOR SHAPIRO WILK TEST,SKEWNESS, KURTOSIS, AND DIFFERENT STATISTICAL
CORRELATION TESTS LIKE ANOVA, PEARSON, SPEARMAN, MUTUAL INFO AND CHI SQ. THIS AIDS IN M.L.PIPELINE AND TO PRODUCE
QUICK RESULTS'''
class stats2:
    def __init__(self,dataframe):
        self.df = dataframe

    '''AUTODETERMINE - DEFINES VARIBALES DATA TYPE'''
    def auto_determine(self,col_name):
        return str(self.df[col_name].dtype)

    '''PRED_OUTCOME- DESIGNATES VARIABLES TO X AND Y, INDEPENDENT AND DEPENDENT VARIABLES, IF THERE IS 
    A PIPELINE WITH DATASETS DEFINED WITH X_ AND Y_VARIABLES'''
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

    '''THIS AGAIN DESIGNATES THE VARIABLES TO INDEPENDENT VARIABLES AND DEPENDENT VARIABLES,BASICALLY
    AS X AND Y, IF THERE IS NO X_ AND Y_ ARE ASSIGNED TO THE VARIABLES'''
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

    '''CHECKS THE VARIABLES DATA DISTRIBUTION AND GIVE SOUT THE MESSAGE WHETHER IT IS DISTRIBUTED OR NO'''
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

    '''GIVES THE MESSAGE OF WHAT TYPE OF KURTOSIS IS A VARIABLE'''
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

    '''GIVES THE MESSAGE OF HOW SKEWED THE VARIABLE IS BASED ON THE ALPHA SET. LESS THE 0 IS LEFT SKEWED,
    BETWEEN 0 AND 1 - FAIRLY SYMMETRICAL, ABOVE 1- RIGHT SKEWED'''
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
            
    '''TEST WHICH IS ALTERNATIE TO T-TEST- WHEN DATA IS NOT NORMALLY DISTRIBUTED'''
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
    
    '''MORE THAN 2 VARIABLES NORMALLY DISTRIBUTED AND NUMERICAL CORRELATION TEST'''
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
        
    '''2 CATEGORICAL DATA VARIABLES'''
    def chi_sq_tst(self,predictor,outcome,alpha = 0.5):
        fun_stats = {"Predictor":predictor,'outcome':outcome,"Testname":"CHI-SQ","Performance stat Raw":None,"keep":False,"comment":" "}
        data_crosstab = pd.crosstab(self.df[predictor],
                            self.df[outcome],
                           margins=True, margins_name="Total")
        _,p,_,_= chi2_contingency(data_crosstab)
        fun_stats["Performance stat Raw"]= p
        if p<=alpha:
            fun_stats["keep"]=True
            fun_stats["comment"]="The variables are correlated, since the pvalue is: " + str(p) + "which is  less than:" + str(alpha)
        else:
            fun_stats["comment"]='The variables are not correlated, since pvalue is:' + str(p) + "which is more than:" + str(alpha)
        return fun_stats

    '''2 NUMERICAL VARIABLES CORRELATION TEST'''
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
        
    '''ONE CATEGORICAL AND ONE NUMERICAL TEST'''
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

    '''2 VARIABLES NUMERICAL AND NOT NORMALLY DISTRIBUTED'''
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

    '''2 CATEGORICAL VARIABLES RELATION TEST WHEN TO IDENTIFY HOW MUCH INFORMATION IS GAINED FROM 
    ONE VARIABLE FROM ANOTHER'''
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

    '''THIS BRINGS ALL THE ABOVE TESTS INTO ONE FUNCTION. BASED ONT HE DATA TYPE AND RELATION BETWEEN THEM
    VARIABLES ARE ASSIGNED TO RESPECTIVE CORRELATION TESTS. FOR EXAMPLE, IF IT IS BETWEEN MORE THAN 2 VARIABLES
    AND NUMERICAL, ANOVA IS ASSIGNED. IN THE END, IT GIVES THE MESSAGE WITH ALL THE INFORMATION ABOUT THE 
    ASSIGNED TEST RESULTS WITH THE MESSAGE'''
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


for_stats = stats2(df)
predict,outcome = for_stats.our_pred_dtypes("classification")
for_stats.main_brain_sys(df,predict,outcome)

catCols = [col for col in final_df.columns if final_df[col].dtype=="O"]
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
for item in catCols:
    final_df[item] = lb_make.fit_transform(final_df[item])

final_df.classification.value_counts()

cat_columns = final_df.select_dtypes(['object']).columns

'''convert all categorical variables to numeric and slicing only the data with CKD = 1'''
final_df[cat_columns] =final_df[cat_columns].apply(lambda x: pd.factorize(x)[0])
final_ckd = final_df[final_df["classification"]== 1]
final_ckd

plt.figure(figsize=(15,8))
df_cor_2 = final_ckd
corr_matrix_2 = df_cor_2.corr()
sns.heatmap(corr_matrix_2, annot=True)
plt.show()


'''APPLYING PCA:
standardized all the data. basically if not standaridized it will take the
larger number column and the principal components will be extracted around that. hence, the dataset needs to
standardized.'''
scaler = StandardScaler()
data3_std = scaler.fit_transform(final_ckd)

'''pca will construct exactly the same number of variables as is in the original dataset, if it is not mentioned.
fit_transform takes variance and sd from the data and standardizes according to these metrics(selection of number of variables).'''
pca = PCA()
pca.fit_transform(data3_std)

'''gives prinicipal components, remember the 1st is larger than followed next larger,descending'''
pca.explained_variance_ratio_

plt.figure(figsize = (20,15))
components = ["component 1", "component 2","component 3", "component 4", "component 5", "component 6", "component 7", "component 8",
             "component 9", "component 10", "component 11","component 12"," component 13","component 14","component 15",
             "component 16", "component 17", "component 18", "component 19","component 20","component 21","component 22",
             "component 23", "component 24", "component 25"]
var_exp = pca.explained_variance_ratio_
plt.bar(components, var_exp)
plt.title("explnd variance by prinicpal variance")
plt.xlabel("Principal components")
plt.ylabel("explnd variance ratio")
plt.show()

'''to see cumulative variance in components. means variance captured until certain component'''
'''below graph explains following:
1 component at 5% variance 8 component= 80%. That is to say inclusive of 1 to 8 components, we have 80%(cumulative).
we need atleast 80% features.(Threshold value). So we pick 8 components.these components are linear combination of initial variables.'''
plt.figure(figsize = (10,6))
plt.plot(range(1,26),pca.explained_variance_ratio_.cumsum(), marker="o", linestyle= "--" )
plt.title("explnd variance by prinicpal variance")
plt.xlabel("Number of components")
plt.ylabel("cumulative explnd variance")
plt.show()

pca = PCA(n_components = 8)

pca.fit(data3_std)

pca.components_

df_pca_comp = pd.DataFrame(data = pca.components_,
                          columns = final_df.columns.values,
                           index = ["component 1","component 2"," component 3", "component 4","component 5",
                                   "component 6", "component 7", "component 8"])
df_pca_comp #CORELATION COEFFICIENTS BY DEFINITIONTAKE VALUES BETWEEN -1 AND +1

plt.figure(figsize=(15,8))
sns.heatmap(df_pca_comp,
           vmin=-1,
           vmax=1,
           cmap='RdBu',
           annot=True)
plt.yticks([0,1,2,3,4,5,6,7],
          ['Component 1','Component 2','Component 3','Component 4',"Component 5",
          "Component 6", "Component 7", "Component 8"],
          rotation=45,
          fontsize=9)

plt.show()

'''As you can see, there is more presence of  blue boxes in component 1 and it reduces 
as they go low in the components'''

pca.transform(data3_std)#produc kmean out of this

'''extract the columns based on the stats,heat map,PCA'''
df_mod = final_df[["age","sg","bu","sc","sod","pot","hemo","pcv","wc","rc","classification"]]
df_mod

X = np.asarray(df_mod.iloc[:,:-1])
y = np.asarray(df_mod.iloc[:,-1])

counter = Counter(y)
#plotting pie chart of distrbution
plt.pie([counter[0],counter[1]],labels=['nockd','ckd'])

'''we will be using svm which is great in handling imbalance dataset and also for the small datasets.
SVM classifies at accurate hyperplane which differenctiate classes. Class_weight assigns weights 
to the features according to their imortance, that way imbalance is handled. Usually, all features
are given equal importance'''

model = SVC(class_weight='balanced')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean Accuracy: %.3f' % mean(scores))

'''lets see how its seems if we do not handle the imbalance aspect'''
model = SVC(gamma='scale')
# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean Accuracy: %.3f' % mean(scores))

'''There is more than 1 percent difference between handling imbalance and not handling imbalance'''

'''how about trying decision tree before handling imbalance'''
model = DecisionTreeClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))

'''After handling imbalance'''
model = DecisionTreeClassifier(class_weight='balanced')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))

'''as we can see there is an evident improvement after handling imbalance'''

X = df_mod.iloc[:,:-1]
y = df_mod.iloc[:,-1]

'''lets continue with full fledge assessment of the dataset'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train.shape, X_test.shape

cols = X_train.columns
X_train = pd.DataFrame(X_train, columns= cols)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_test = pd.DataFrame(X_test, columns=[cols])

X_train = pd.DataFrame(X_train, columns=[cols])

X_train.describe()

# instantiate classifier with default hyperparameters
svc=SVC(class_weight = "balanced")

# fit classifier to training set
svc.fit(X_train,y_train)

# make predictions on test set
y_pred=svc.predict(X_test)

# compute and print accuracy score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

'''compare the train-set and test-set accuracy '''
y_pred_train = svc.predict(X_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

y_pred_test=svc.predict(X_test)

'''checking overfitting - no overfitting since train and test score almost comparable'''
print('Training set score: {:.4f}'.format(svc.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(svc.score(X_test, y_test)))

y_test.value_counts()

'''confusion matrix'''
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test))

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# print classification accuracy


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

# print precision score
'''Precision is the percentage of correctly 
predicted positive outcomes out of all the predicted positive outcomes. It is more focused
on positive class than the negative class.'''

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))

'''Recall is the percentage of correctly predicted 
positive outcomes out of all the actual positive outcomes'''
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))

'''True Positive Rate is nothing but Recall.'''
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


'''false rate'''
false_positive_rate = FP / float(FP + TN)

print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))

# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Predicting a CKD classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred_test)

print('ROC AUC : {:.4f}'.format(ROC_AUC))

# calculate cross-validated ROC AUC

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(svc, X_train, y_train, cv=10, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


'''Roc and Auc with cross validation proves that svm is an excellent classifier.'''
