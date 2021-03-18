#stage 1 

#importing the useful libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## importing the file

hr_train = pd.read_csv("aug_train.csv")
hr_test = pd.read_csv("aug_test.csv")

hr_train = pd.DataFrame(hr_train)
hr_test = pd.DataFrame(hr_test)

#Data pre processing

# stage 2
hr_train.info

hr_train.dtypes
hr_train.shape
hr_test.shape

desc_num = hr_train.describe()
desc_ojject = hr_train.describe(include="object")
hr_test.describe()

hr_train.dtypes
##**** by looking at the info, we observed that hr_test does not have the target variable
##**** city_development_index is already scaled
##**** datatype of few variables needs to be changes 

#missing data

hr_train.isnull().sum()
plt.figure(figsize=(20,8))
sns.heatmap(hr_train.isnull(),yticklabels=False)
#from the above heatmap and isnull code, we observed that 4 variables namely
#"gender","major_discipline","company_size" & "company_type" have many missing values.

## ********________EDA for missing value treatment_______*********** 

flatui = ["#1f4068","#e43f5a","#ff6768","#6b778d"]
sns.set_palette(flatui)
sns.palplot(sns.color_palette())
exp = hr_train.copy()
plt.style.available
plt.style.use("fivethirtyeight")


## univariate analysis
exp.city.value_counts()
exp.city.unique()
sns.countplot(exp.city)
plt.xticks(rotation=90)
sns.countplot(exp.gender)
sns.countplot(exp.relevent_experience)#around 14000 candidates has relevent experience
sns.countplot(exp.enrolled_university)# only 4000 candidates enrolled in a full time course
sns.countplot(exp.education_level)# most of the candidates are graduate, very few hold a phd degree
sns.countplot(exp.major_discipline)# most of the candidates are from stem background
plt.xticks(rotation=35)

sns.countplot(y="company_type", data = exp)
plt.xticks(rotation=45)
# most of the candidates opt for Pvt. companies and public sector comes 2nd
sns.countplot(exp.last_new_job)
# most candidates has a gap of least 1 year, around 3100 candidates has more than 4 years
sns.histplot(x='training_hours', data = exp)
# training hours is left skewed which means



#multivariate analysis


exp.gender.value_counts().sum()
gender = exp[exp['target']==1]['gender']
gender.value_counts().sum()
sns.countplot(gender)
sns.countplot(x='gender',hue='target', data = exp)
## around 3388 people are looking for a new job, out of which 3012 are males

sns.countplot(x='education_level',hue='gender', data = exp)
#most of the graduates and master holders are males 

sns.countplot(x='education_level',hue='target', data = exp)
# most of graduates are looking for a job change

sns.countplot(x='gender',hue='major_discipline', data = exp)
# most of the candidates from stem background are males

exp.training_hours.max()
exp.training_hours.mean()
sns.pairplot(exp, hue='target')
sns.heatmap(exp.corr())
## from the above correlation plot we can observe that there is no correlation between the dependent & independent variable

##******________ filling missing data_________*******

# missing values
perc_miss = exp.isnull().sum()/exp.shape[0]*100
print(perc_miss)
## company_size and company type has the most missing values

# filling missing values with mode
col_mode = ['gender','enrolled_university','education_level','major_discipline']
for col in col_mode:
    exp[col].fillna(exp[col].mode()[0],inplace=True)
exp.isnull().sum()
# as we can see that there are no missing values left in the above columns

#1. changing the data types

#experience

#bfore changing datatype of experience, we changed the values '>20' to '21' & '<1' to '0'
exp['experience'] = exp['experience'].replace('>20','21')
exp['experience'] = exp['experience'].replace('<1','0')

exp['experience'] = exp['experience'].astype("float")
exp['experience'].describe()
sns.histplot(exp.experience)
exp.dtypes
exp.experience.value_counts()
#we will replace missing values in experience with mode
exp.experience.mode()
exp['experience'].replace(np.nan,21,inplace=True)
exp.isnull().sum()#no missing value in experience

# last new job
exp['last_new_job'] = exp['last_new_job'].replace('>4','5')
exp['last_new_job'] = exp['last_new_job'].replace('never','0')


exp['last_new_job'] = exp['last_new_job'].astype("float")
exp['last_new_job'].describe()
exp.dtypes

sns.countplot(exp.last_new_job)
exp.last_new_job.median()
exp.last_new_job.mode()
exp.last_new_job.mean()
# we will replace missing values with 1 (mode & median value)
exp['last_new_job'].replace(np.nan,1,inplace=True)
exp.isnull().sum()#no missing value in experience

#### to replace missing values in company_type and company_size, we will Replace NAN categories with most occurred values, and add a new feature to introduce some weight/importance to non-imputed and imputed observations.

# Function to impute most occured category and add importance vairable
def impute_nan_add_vairable(DataFrame,ColName):
    #1. add new column and replace if category is null then 1 else 0
    DataFrame[ColName+"_Imputed"] =   np.where(DataFrame[ColName].isnull(),1,0)
    
    # 2. Take most occured category in that vairable (.mode())
    
    Mode_Category = DataFrame[ColName].mode()[0]
    
    ## 2.1 Replace NAN values with most occured category in actual vairable
    
    DataFrame[ColName].fillna(Mode_Category,inplace=True)
# Call function to impute NAN values and add new importance feature
for Columns in ['company_type','company_size']:
    impute_nan_add_vairable(exp,Columns)
    
# Display top 10 row to see the result of imputation
exp[['company_type','company_size','company_type_Imputed','company_size_Imputed']].head(10)

exp.isnull().sum()
## as you can see there is no missing value left in the dataframe
exp.shape



#### more EDA ####

city_dev1 = exp[exp['target']==1]['city_development_index']
city_dev0 = exp[exp['target']==0]['city_development_index']
city_dev1.value_counts().sum()
city_dev0.value_counts().sum()
sns.distplot(city_dev1, hist=False)
#the above plot is quite interesting, as there are 2 stages where the data peaks, certainly at 0.6 & 0.9
# from this we can conclude that perhaps people in cities with 0.9 development rate look for jobs because of better opportunities
# and people in cities with 0.6 development index are looking for opportunities to grow and improve
sns.distplot(city_dev0, hist=False)

sns.lineplot('experience', 'training_hours', data=exp)

sns.lineplot('company_type', 'training_hours', data=exp)
plt.xticks(rotation=45)





# Statistical tests
## performing chisquare test on categrical variables

from scipy.stats import chi2_contingency 

# 1. target vs city

contigency= pd.crosstab(exp['city'], exp['target'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 rejected as p value is <0.5, important variable 




# 2.taget vs gender

contigency= pd.crosstab(exp['gender'], exp['target'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 rejected as p value is <0.05, not an important variable 



# 3.target vs relevent_experience
contigency= pd.crosstab(exp['target'], exp['relevent_experience'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 rejected as p value is <0.05, important variable 
    
 
       
# 4.target vs enrolled_university
exp.enrolled_university.value_counts()
'''
no_enrollment       14203
Full time course     3757
Part time course     1198
Name: enrolled_university, dtype: int64
'''

'''
we will make two categories
'''
exp['enrolled_university']=exp.get('enrolled_university').replace('Full time course','enrolled')
exp['enrolled_university']=exp.get('enrolled_university').replace('Part time course','enrolled')

exp.enrolled_university.value_counts()# new values
sns.countplot(exp.enrolled_university)## new plot with new categories

contigency= pd.crosstab(exp['target'], exp['enrolled_university'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 rejected as p value is <0.05, important variable     
    




# 5.target vs education level

exp.education_level.value_counts()
'''
Graduate          12058
Masters            4361
High School        2017
Phd                 414
Primary School      308
Name: education_level, dtype: int64

we will merge masters & phd as PG, high school & primary scholl as school
'''

exp['education_level']=exp.get('education_level').replace('Masters','PG')
exp['education_level']=exp.get('education_level').replace('Phd','PG')
exp['education_level']=exp.get('education_level').replace('High School','School')
exp['education_level']=exp.get('education_level').replace('Primary School','School')

exp.education_level.value_counts()# new values with new categories
sum(exp.education_level.value_counts())
sns.countplot(exp.education_level)# new plot with new categories

contigency= pd.crosstab(exp['target'], exp['education_level'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 rejected as p value is <0.05, important variable   




# 6.target major_discipline

exp.major_discipline.value_counts()
'''
STEM               17305
Humanities           669
Other                381
Business Degree      327
Arts                 253
No Major             223
Name: major_discipline, dtype: int64

we will make two categories, STEM & others
'''

exp['major_discipline']=exp.get('major_discipline').replace('Business Degree','Other')
exp['major_discipline']=exp.get('major_discipline').replace('Arts','Other')
exp['major_discipline']=exp.get('major_discipline').replace('Humanities','Other')
exp['major_discipline']=exp.get('major_discipline').replace('No Major','Other')

exp.major_discipline.value_counts()# new values with new categories
sum(exp.major_discipline.value_counts())
sns.countplot(exp.major_discipline)# new plot with new categories

contigency= pd.crosstab(exp['target'], exp['major_discipline'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 rejected as p value is >0.05, not an important variable   




# 7.target vs company_size

exp.company_size.value_counts()
'''
50-99        9021
100-500      2571
10000+       2019
10/49        1471
1000-4999    1328
<10          1308
500-999       877
5000-9999     563
Name: company_size, dtype: int6
'''
contigency= pd.crosstab(exp['target'], exp['company_size'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 rejected as p value is <0.05, important variable   




#8 8.target vs company_type

exp.company_type.value_counts()
'''
Pvt Ltd                15957
Funded Startup          1001
Public Sector            955
Early Stage Startup      603
NGO                      521
Other                    121
Name: company_type, dtype: int64

we will make 2 categories, Pvt Ltd and Others
'''
exp['company_type']=exp.get('company_type').replace('Funded Startup','Other')
exp['company_type']=exp.get('company_type').replace('Early Stage Startup','Other')
exp['company_type']=exp.get('company_type').replace('Public Sector','Other')
exp['company_type']=exp.get('company_type').replace('NGO','Other')

contigency= pd.crosstab(exp['target'], exp['company_type'])
contigency

exp.company_type.value_counts()#new values
sum(exp.company_type.value_counts())
sns.countplot(exp.company_type)#new plot

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 accepted as p value is <0.05, important variable   




## 9.target vs last_new_job

exp.last_new_job.value_counts()
'''
1.0    8463
5.0    3290
2.0    2900
0.0    2452
4.0    1029
3.0    1024
Name: last_new_job, dtype: int64
'''
contigency= pd.crosstab(exp['target'], exp['last_new_job'])
contigency

c, p, dof, expected = chi2_contingency(contigency)
p

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 
## H0 accepted as p value is <0.05, important variable   






## Performing t-test for continous variables
from scipy import stats

#1. target vs city_development_index

CDI_1= exp[exp.target== 1]
CDI_0= exp[exp.target== 0]
import scipy
scipy.stats.ttest_ind(CDI_1.city_development_index, CDI_0.city_development_index)

'''
Ttest_indResult(statistic=-50.31616259328961, pvalue=0.0)
'''
p=0.0

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)')
# H0 rejected as p value is <0.05, important variable   




# 2.target vs training_hours

TH_1= exp[exp.target== 1]
TH_0= exp[exp.target== 0]
len(TH_1)
len(TH_0)

statistics, p=scipy.stats.ttest_ind(TH_1.training_hours, TH_0.training_hours)
p
'''
Ttest_indResult(statistic=-2.9870990541592386, pvalue=0.002819949452636266)
'''

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)')
# H0 rejected as p value is <0.05, important variable




# 3. target vs experience

EX_1= exp[exp.target== 1]
EX_0= exp[exp.target== 0]
len(EX_1)
len(EX_0)

statistics, p=scipy.stats.ttest_ind(EX_1.experience, EX_0.experience)
p
'''
Ttest_indResult(statistic=-24.49269196805569, pvalue=1.7868071617356925e-130)
'''

alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)')
# H0 rejected as p value is <0.05, important variable


'''
after doing chisquare test and independent t-test on variables we have decided to remove
"major_discipline", "enrolled_id" & "gender" for making the model

'''


#####*********** MODEL1 ***********#########

# Feature scaling

from sklearn.preprocessing import StandardScaler
std_scale= StandardScaler()
exp['experience'] = std_scale.fit_transform(exp[['experience']])
exp['last_new_job'] = std_scale.fit_transform(exp[['last_new_job']])
exp['training_hours'] = std_scale.fit_transform(exp[['training_hours']])
exp[['experience','last_new_job','city_development_index','training_hours']].head(3)

# Encoding
df = exp.copy()
dumies = pd.get_dummies(df[['relevent_experience','enrolled_university',
                            'education_level','company_size','company_type']])
dumies.head(3)

df = df.drop(columns=['relevent_experience','enrolled_university',
                      'education_level','company_size','company_type'])

df = df.drop(columns=['city']) # droping city because it has 123 values, which can make our model expensive when encoded
df = df.drop(columns=['gender','enrollee_id','major_discipline'])#droping columns with p value <0.05

df= pd.concat([df,dumies], axis=1)
df.head()


X = df.drop(['target'], axis = 1)
Y = df['target']

# balancing the dataset
#our dataset is imbalnces as it can be observed by the plot below

sns.countplot(df.target)

from imblearn.over_sampling import SMOTE
# OverSampling using SMOTE
smote = SMOTE(random_state = 402)
X_smote, Y_smote = smote.fit_resample(X,Y)

print(X_smote.shape, Y_smote.shape)

from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(Y_smote)))

#After re-sampling
df = pd.concat([X_smote, Y_smote], axis=1)
sns.countplot(df.target)
df.target.value_counts()
df.target.shape


# spliting the data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_smote, Y_smote, test_size = 0.2 ,random_state = 42)





##***** Logistic Regression ******##

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
from sklearn.metrics import confusion_matrix

# A parameter grid for Logistic Regression
params = {
        'C': [0.001,0.01,0.1,1,10],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
         }

from sklearn.model_selection import RandomizedSearchCV

random_cv=RandomizedSearchCV(estimator=LR,param_distributions=params,
                             cv=5,n_iter=5,scoring='roc_auc',n_jobs=1,verbose=3,return_train_score=True,random_state=121)
random_cv.fit(X_train,y_train)

#best parameter 
random_cv.best_params_

#Model Building
LR = LogisticRegression(C = 1, solver = 'sag').fit(X_train,y_train)
LR

#Prediction & Evaluation

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
yhat[0:5]
yhat_prob[0:5]

from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat)

#Confusion matrix plotting
from sklearn.metrics import confusion_matrix
labels = ['will take', 'wont take']
cm=confusion_matrix(y_test, yhat)
cm
axes=sns.heatmap(cm, square=True, annot=True,fmt='d',cbar=True,cmap=plt.cm.Blues)
ticks=np.arange(len(labels))+0.5
plt.title('Confusion matrix of the classifier')
plt.xlabel('True')
plt.ylabel('Predicted')
axes.set_xticks(ticks)
axes.set_xticklabels(labels,rotation=0)
axes.set_yticks(ticks)
axes.set_yticklabels(labels,rotation=0)


from sklearn.metrics import classification_report
print (classification_report(y_test, yhat))



#different accuracy scores
from sklearn.metrics import log_loss
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
print("Logistic Regression's Accuracy: ", metrics.accuracy_score(y_test, yhat))
print("Logistic Regression's LogLoss : ", log_loss(y_test, yhat_prob))
print("Logistic Regression's F1-Score: ", f1_score(y_test, yhat, average='weighted'))
-np.mean(cross_val_score(LR,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 4))


#ROC curve
#!pip install scikit-plot
import scikitplot as skplt
y_true = y_test
y_probas = yhat_prob
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()

'''
Logistic regression score Logistic Regression's Accuracy:  0.7632539544585434
AUC  : 0.82

Confusion Matrix
[2098,  779]
[ 583, 2293]

Classification Report
              precision    recall  f1-score   support

         0.0       0.78      0.73      0.75      2877
         1.0       0.75      0.80      0.77      2876

    accuracy                           0.76      5753
   macro avg       0.76      0.76      0.76      5753
weighted avg       0.76      0.76      0.76      5753
0.76 seems a good accuracy score, although model can be tuned further to achieve a higer accuracy!
'''




##******DECISION TREE*******####

from sklearn.tree import DecisionTreeClassifier

# A parameter grid for Logistic Regression

params = {
        'criterion': ["gini", "entropy"],
        'splitter': ["best","random"],
        'max_depth': range(1,30),
        'max_features':["auto", "sqrt", "log2"]
         }

DT= DecisionTreeClassifier()
DT# it shows the default parameters

random_cv=RandomizedSearchCV(estimator=DT,param_distributions=params,
                             cv=5,n_iter=5,scoring='roc_auc',n_jobs=1,verbose=3,return_train_score=True,random_state=121)
random_cv.fit(X_train,y_train)

random_cv.best_params_

# Model Building
DT= DecisionTreeClassifier(criterion = 'entropy', splitter ='best',
                           max_depth= 18).fit(X_train,y_train)
DT


#Prediction & Evaluation


yhat = DT.predict(X_test)
yhat_prob = DT.predict_proba(X_test)
yhat[0:5]
yhat_prob[0:5]

yhat_prob = DT.predict_proba(X_test)
yhat_prob


jaccard_score(y_test, yhat)

#Confusion matrix plotting
labels = ['will take', 'wont take']
cm=confusion_matrix(y_test, yhat)
cm
axes=sns.heatmap(cm, square=True, annot=True,fmt='d',cbar=True,cmap=plt.cm.Blues)
ticks=np.arange(len(labels))+0.5
plt.title('Confusion matrix of the classifier')
plt.xlabel('True')
plt.ylabel('Predicted')
axes.set_xticks(ticks)
axes.set_xticklabels(labels,rotation=0)
axes.set_yticks(ticks)
axes.set_yticklabels(labels,rotation=0)


#Classification Report
print (classification_report(y_test, yhat))



#different accuracy scores

print("Decision Tree's Accuracy: ", round(metrics.accuracy_score(y_test, yhat),2))
print("Decision Tree's LogLoss : ", log_loss(y_test, yhat_prob))
print("Decision Tree's F1-Score: ", round(f1_score(y_test, yhat, average='weighted'),2))
-np.mean(cross_val_score(DT,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 4))


#ROC curve

y_true = y_test
y_probas_tree = DT.predict_proba(X_test)
skplt.metrics.plot_roc(y_true, y_probas_tree)
plt.show()

'''
Classification Report
       precision    recall  f1-score   support

         0.0       0.81      0.79      0.80      2877
         1.0       0.79      0.81      0.80      2876

    accuracy                           0.80      5753
   macro avg       0.80      0.80      0.80      5753
weighted avg       0.80      0.80      0.80      5753

Confusion Matrix
[2259,  618]
[536, 2340]

Decision Tree's Accuracy:  0.7994090039979141

Decision Tree's LogLoss :  4.586660462344769

Decision Tree's F1-Score:  0.7993687408562533

Cross_val_Score: 0.21204769320330422
 
AUC = 0.82
'''

"""
Logistic regression predicted with an accuracy of 76% and Decision Tree wit an 
overall accuracy of 80%, although AUC of both the models remained the same.
Hence, we will go with the Decision Tree model.
"""
