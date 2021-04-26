#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 


# In[21]:


# read the data
df = pd.read_excel('C:\\Users\\123\\Desktop\\dataset.xlsx')


# In[22]:


# checking first thirty rows of our dataset
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
df.head(30)


# In[23]:


df.shape


# In[24]:


(df.isna().sum()/df.shape[0]).sort_values(ascending=True)


# In[25]:


#data cleaning- drop columns whose missing rate >99%
df = df[df.columns[df.isna().sum()/df.shape[0] <0.99]]
df.head()


# In[26]:


df.shape


# In[27]:


#data evaluation
df['SARS-Cov-2 exam result'].value_counts(normalize=True)


# In[28]:


#positive/negative by each index
for col in df.select_dtypes('object'):
    print(df[col].value_counts())


# In[29]:


sns.countplot(x='Patient age quantile', hue='SARS-Cov-2 exam result',data=df)


# In[30]:


positive_df = df[df['SARS-Cov-2 exam result'] =='positive']
negative_df = df[df['SARS-Cov-2 exam result'] =='negative']


# In[31]:


missing_rate = df.isna().sum()/df.shape[0]
blood_columns = df.columns[(missing_rate<0.9)&(missing_rate> 0.88)]
viral_columns = df.columns[(missing_rate<0.88)&(missing_rate> 0.75)]


# In[32]:


#positive/negative graph by each index
for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col],label='positive')
    sns.distplot(negative_df[col],label='negative')
    plt.legend()


# In[33]:


#ward distribution graph by each index
def hospitalization(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'semi intensive care'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'intensive care'
    else:
        return 'unknown'
df['status'] = df.apply(hospitalization,axis=1)
for col in blood_columns:
    plt.figure()
    for cat in df['status'].unique():
        sns.distplot(df[df['status']==cat][col],label=cat)
    plt.legend()


# In[34]:


#KNN


# In[35]:


#replace positives to 1, and negatives to 0
df.replace('not_detected', 0, inplace=True)
df.replace('detected', 0, inplace=True)
df.replace('absent', 0, inplace=True)
df.replace('present', 1, inplace=True)
df.replace('negative', 0, inplace=True)
df.replace('positive', 1, inplace=True)
# replace NaNs by 0
df = df.fillna(0)


# In[36]:


#splitting train and test
# for splitting data into training and testing data
from sklearn.model_selection import train_test_split
# defining target variables 
target = df['SARS-Cov-2 exam result']

# defining predictor variables 
features = df.select_dtypes(exclude=[object])

# assigning the splitting of data into respective variables
X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify = target)


# In[37]:


print("Number of samples in train set: %d" % y_train.shape)
print("Number of positive samples in train set: %d" % (y_train == 1).sum(axis=0))
print("Number of negative samples in train set: %d" % (y_train == 0).sum(axis=0))
print()
print("Number of samples in test set: %d" % y_test.shape)
print("Number of positive samples in test set: %d" % (y_test == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (y_test == 0).sum(axis=0))


# In[38]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
# assigning the dictionary of variables whose optimium value is to be retrieved
param_grid = {'n_neighbors' : np.arange(1,50)}
# performing Grid Search CV on knn-model, using 5-cross folds for validation of each criteria
knn_cv = GridSearchCV(knn, param_grid, cv=5)
# training the model with the training data and best parameter
knn_cv.fit(X_train,y_train)


# In[39]:


# finding out the best parameter chosen to train the model
print("The best paramter we have is: {}" .format(knn_cv.best_params_))

# finding out the best score the chosen parameter achieved
print("The best score we have achieved is: {}" .format(knn_cv.best_score_))


# In[40]:


# predicting the values using the testing data set
y_pred = knn_cv.predict(X_test)


# In[41]:


# the score() method allows us to calculate the mean accuracy for the test data
print("The score accuracy for training data is: {}" .format(knn_cv.score(X_train,y_train)))
print("The score accuracy for testing data is: {}" .format(knn_cv.score(X_test,y_test)))
# for performance metrics
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
# call the classification_report and print the report
print(classification_report(y_test, y_pred))
# call the confusion_matrix and print the matrix
print(confusion_matrix(y_test, y_pred))


# In[42]:


#naive bayes


# In[43]:


# overall libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from collections import OrderedDict
from IPython.core.pylabtools import figsize
import re

# plotting libraries
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
get_ipython().run_line_magic('matplotlib', 'inline')

# sklearn libraries
import sklearn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix, average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer


# In[44]:


pd.set_option("max_columns", None)
pd.set_option("max_rows", None)


# In[45]:


# Checking the unique values for the SARS-Cov-2 exam result

df['SARS-Cov-2 exam result'].unique()


# In[46]:


# read the data
df = pd.read_excel('C:\\Users\\123\\Desktop\\dataset.xlsx')
df.head()


# In[47]:


# Checking the unique values for the SARS-Cov-2 exam result

df['SARS-Cov-2 exam result'].unique()


# In[48]:


# Replacing negative to 0 an positive to 1 and then checking if it worked

df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].replace({'negative': 0, 'positive': 1})
df['SARS-Cov-2 exam result'].unique()


# In[49]:


# checking the categorical variables

df.select_dtypes(include = ['object']).columns


# In[50]:


# replacing the values to make them numerical, I am doing them by hand to make sure all the exams make sense.
# This is possible because there aren't many categorical variables.
# To do this, I checked all the variables unique values and created an unique dictionary

df.loc[:,'Respiratory Syncytial Virus':'Parainfluenza 2'] = df.loc[:,'Respiratory Syncytial Virus':'Parainfluenza 2'].replace({'not_detected':0, 'detected':1})
df.loc[:,'Influenza B, rapid test':'Strepto A'] = df.loc[:,'Influenza B, rapid test':'Strepto A'].replace({'negative':0, 'positive':1})
df['Urine - Esterase'] = df['Urine - Esterase'].replace({'absent':0})
df['Urine - Aspect'] = df['Urine - Aspect'].replace({'clear':0, 'cloudy':2, 'altered_coloring':3, 'lightly_cloudy':1})
df['Urine - pH'] = df['Urine - pH'].replace({'6.5':6.5, '6.0':6.0,'5.0':5.0, '7.0':7.0, '5':5, '5.5':5.5,
       '7.5':7.5, '6':6, '8.0':8.0})
df['Urine - Hemoglobin'] = df['Urine - Hemoglobin'].replace({'absent':0, 'present':1})
df.loc[:,'Urine - Bile pigments':'Urine - Nitrite'] = df.loc[:,'Urine - Bile pigments':'Urine - Nitrite'].replace({'absent':0})
df.loc[:,'Urine - Urobilinogen':'Urine - Protein'] = df.loc[:,'Urine - Urobilinogen':'Urine - Protein'].replace({'absent':0, 'normal':1})
df['Urine - Hemoglobin'] = df['Urine - Hemoglobin'].replace({'absent':0, 'present':1, 'not_done':np.nan})
df['Urine - Leukocytes'] = df['Urine - Leukocytes'].replace({'38000':38000, '5942000':5942000, '32000':32000, '22000':22000,'<1000': 900, '3000': 3000,'16000':16000, '7000':7000, '5300':5300, '1000':1000, '4000':4000, '5000':5000, '10600':106000, '6000':6000, '2500':2500, '2600':2600, '23000':23000, '124000':124000, '8000':8000, '29000':29000, '2000':2000,'624000':642000, '40000':40000, '3310000':3310000, '229000':229000, '19000':19000, '28000':28000, '10000':10000,'4600':4600, '77000':77000, '43000':43000})
df['Urine - Crystals'] = df['Urine - Crystals'].replace({'Ausentes':0, 'Urato Amorfo --+':1, 'Oxalato de Cálcio +++':3,'Oxalato de Cálcio -++':2, 'Urato Amorfo +++':4})
df.loc[:,'Urine - Hyaline cylinders':'Urine - Yeasts'] = df.loc[:,'Urine - Hyaline cylinders':'Urine - Yeasts'].replace({'absent':0})
df['Urine - Color'] = df['Urine - Color'].replace({'light_yellow':0, 'yellow':1, 'orange':2, 'citrus_yellow':1})
df = df.replace('not_done', np.NaN)
df = df.replace('Não Realizado', np.NaN)


# In[51]:


# Dropping the patient ID column

df = df.drop('Patient ID', axis = 1)


# In[52]:


# checking if all of the categorical variables were treated

df.select_dtypes(include = ['object']).columns


# In[53]:


# let's create a rank of missing values

null_count = df.isnull().sum().sort_values(ascending=False)
null_percentage = null_count / len(df)
null_rank = pd.DataFrame(data=[null_count, null_percentage],index=['null_count', 'null_ratio']).T
null_rank


# In[54]:


# dropping columns that don't have any content in it

df = df.drop(['Mycoplasma pneumoniae','Urine - Nitrite', 'Urine - Sugar',  'Prothrombin time (PT), Activity', 'D-Dimer'], axis = 1)


# In[55]:


# filling missing values with 0

df[['Urine - Leukocytes', 'Urine - pH']] = df[['Urine - Leukocytes', 'Urine - pH']].fillna(0)


# In[56]:


# filling missing values with -1

df[['Patient age quantile', 'SARS-Cov-2 exam result', 'Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2', 'Influenza B, rapid test', 'Influenza A, rapid test', 'Strepto A', 'Fio2 (venous blood gas analysis)','Myeloblasts', 'Urine - Esterase', 'Urine - Hemoglobin', 'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Protein', 'Urine - Crystals', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']] = df[['Patient age quantile', 'SARS-Cov-2 exam result', 'Respiratory Syncytial Virus', 'Influenza A', 'Influenza B', 'Parainfluenza 1', 'CoronavirusNL63', 'Rhinovirus/Enterovirus', 'Coronavirus HKU1', 'Parainfluenza 3', 'Chlamydophila pneumoniae', 'Adenovirus', 'Parainfluenza 4', 'Coronavirus229E', 'CoronavirusOC43', 'Inf A H1N1 2009', 'Bordetella pertussis', 'Metapneumovirus', 'Parainfluenza 2', 'Influenza B, rapid test', 'Influenza A, rapid test', 'Strepto A', 'Fio2 (venous blood gas analysis)','Myeloblasts', 'Urine - Esterase', 'Urine - Hemoglobin', 'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Protein', 'Urine - Crystals', 'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts', 'Urine - Color']].fillna(-1)


# In[57]:


# filling all the other missing values with 99

df = df.fillna(99)


# In[58]:


# let's see if there is still any missing values left

null_count = df.isnull().sum().sort_values(ascending=False)
null_percentage = null_count / len(df)
null_rank = pd.DataFrame(data=[null_count, null_percentage],index=['null_count', 'null_ratio']).T
null_rank


# In[59]:


# creating a scaler and using it, disconsidering the target column

scaler = MinMaxScaler()
addmits = pd.DataFrame(df[['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']], columns = ['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'])
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop(['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis = 1)), columns = (df.drop(['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'], axis = 1).columns))


# In[60]:


# concatenating all the columns again

df_total = pd.concat([addmits, df_scaled], axis = 1)


# In[61]:


# checking if the concatening worked

df_total.head()


# In[62]:


# renaming the columns with backslash

df_total = df_total.rename(columns={"Meancorpuscularhemoglobinconcentration\xa0MCHC": "Meancorpuscularhemoglobinconcentrationxa0MCHC", "Gammaglutamyltransferase\xa0": "Gammaglutamyltransferasexa0", "Ionizedcalcium\xa0": "Ionizedcalciumxa0", "Creatinephosphokinase\xa0CPK\xa0" : "Creatinephosphokinasexa0CPKxa0"})


# In[63]:


# Let's remove all special characters and spaces from the column names
# We will also make them lowercase

df_total.columns=df_total.columns.str.replace(r'\(|\)|:|,|;|\.|’|”|“|\?|%|>|<|(|)','')
df_total.columns=df_total.columns.str.replace(r'/','')
df_total.columns=df_total.columns.str.replace(' ','')
df_total.columns=df_total.columns.str.replace('"','')
df_total.columns=df_total.columns.str.replace('\-','')
df_total.columns=df_total.columns.str.replace('\=','')
df_total.columns=df_total.columns.str.replace('\#','')
df_total.columns=df_total.columns.str.lower()


# In[64]:


# removing all the lines that don't give out a positive result for the sars-cov2 exam

df_total = df_total[df_total['sarscov2examresult'] != 0]


# In[65]:


# the first column will remain with the 1 value, the other two will be replaced with 2 and 3

df_total['patientaddmitedtosemiintensiveunit1yes0no'] = df_total['patientaddmitedtosemiintensiveunit1yes0no'].replace({1:2})
df_total['patientaddmitedtointensivecareunit1yes0no'] = df_total['patientaddmitedtointensivecareunit1yes0no'].replace({1:3})


# In[66]:


# creating one single column that sums up all the directions to where the patients where sent

df_total['patient'] = df_total.apply(lambda row: row.patientaddmitedtoregularward1yes0no + row.patientaddmitedtosemiintensiveunit1yes0no + row.patientaddmitedtointensivecareunit1yes0no, axis=1)
df_total.head()


# In[67]:


# dropping the first three columns

df_total = df_total.drop(['patientaddmitedtoregularward1yes0no', 'patientaddmitedtosemiintensiveunit1yes0no', 'patientaddmitedtointensivecareunit1yes0no'], axis = 1)


# In[77]:


# Creating X and y

X = df_total.drop(['patient'], axis = 1)
y = df_total['patient']


# In[84]:


# let's split the X and y into a test and train set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = 42)


# In[85]:


# Creating a Gaussian Naive Bayes Classifier

gnb = GaussianNB().fit(X_train, y_train) 
y_pred = gnb.predict(X_test)


# In[86]:


# Calculating the score of the model

accuracy = gnb.score(X_test, y_test) 
print(accuracy)


# In[87]:


# Calculating the ROC AUC score of the model

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

multiclass_roc_auc_score(y_test, y_pred)


# In[88]:


# for performance metrics
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix


# In[89]:


# call the classification_report and print the report
print(classification_report(y_test, y_pred))
# call the confusion_matrix and print the matrix
print(confusion_matrix(y_test,y_pred))


# In[ ]:




