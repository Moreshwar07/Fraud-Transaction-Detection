from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_curve,auc, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import f1_score,precision_recall_curve
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
credit_data= pd.read_csv("C:/Users/Admin/PycharmProjects/Fruad_Transaction/archive/credit_card.csv")
import copy
credit_data= copy.deepcopy(credit_data)
credit_data
#shape of the data
credit_data.shape
credit_data.columns
credit_data.info()
#check for missing value
credit_data.isnull().sum()
#check for duplicates
sum(credit_data.duplicated())
# check for zeros in columns
#print(credit_data[credit_data==0].count())
#print("Total % of Zeros as Value in Columns")
credit_data[credit_data==0].count()/credit_data.shape[0]*100
descrete_feature = [feature for feature in credit_data if len(credit_data[feature].unique()) < 25]
#print("Descrete Variables Count: {}".format(len(descrete_feature)), "\n")

for feature in descrete_feature:
    print('The feature is {} and number of numerical are {}'.format(feature, len(credit_data[feature].unique())))

for feature in descrete_feature:
    print("\n", feature, " : ", credit_data[feature].unique())
continuous_feature=[feature for feature in credit_data if feature not in descrete_feature ]
print("Continuous Variables Count {}".format(len(continuous_feature)),"\n")

for feature in continuous_feature:
    print('The feature is:  {}     and number of numerical are:  {}'.format(feature,len(credit_data[feature].unique())))
#value_count for class
class_valuecount= credit_data['Class'].value_counts()
print(class_valuecount)
print("percentage value for class==0 are Not_Fraudulent ")
print(round(class_valuecount[0]/len(credit_data)*100,2),"%")
print("percentage value for class==1 are Fraudulent ")
print(round(class_valuecount[1]/len(credit_data)*100,2),"%")
sns.set_theme(style="darkgrid")
sns.countplot(x= "Class", data= credit_data)
plt.title("Class count_plot")
plt.show
summary_1 = (credit_data[credit_data['Class'] == 1].describe().transpose().reset_index())
summary_1 = summary_1.rename(columns = {"index" : "feature"})
#print("check other variable with respect of class==1")
summary_1
summary_0 = (credit_data[credit_data['Class'] == 0].describe().transpose().reset_index())
summary_0 = summary_0.rename(columns = {"index" : "feature"})
#print("check other variable with respect of class==0 \n summary_0")
summary_0
#sns.scatterplot(x='Time', y='Amount', hue='Class', data=credit_data)
#Lets check data distribution respect of Amount and time
fig, ax= plt.subplots(1,2, figsize= (10,6))

amount= credit_data.Amount.values
time= credit_data.Time.values

sns.distplot(amount, ax= ax[0], color= "darkorange")
ax[0].set_title("Distribution of transaction amount")
ax[0].set_xlim([min(amount), max(amount)])
plt.show


sns.distplot(time, ax= ax[1], color= "forestgreen")
ax[1].set_title("Distribution of transaction Time")
ax[1].set_xlim([min(time), max(time)])
plt.show
credit_data[['Amount','Time']].describe()
#sns.pairplot(credit_data)
# Find out correlation between variables of data
credit_data.corr()
#Plotting a heatmap to visualize the correlation between the variables
fig, ax = plt.subplots(figsize=(15,8))
sns.heatmap(credit_data.corr(), ax=ax)
#lets check more about amount variable
credit_data[(credit_data['Class']==1)] ["Amount"].value_counts().head(10)
# converting seconds to time delta to extract hours and mins

timedelta = pd.to_timedelta(credit_data['Time'], unit='s')

credit_data['mins'] = (timedelta.dt.components.minutes).astype(int)
credit_data['hours'] = (timedelta.dt.components.hours).astype(int)
sns.scatterplot(x='Time', y='Amount', hue='Class', data=credit_data)
fig,axs= plt.subplots(3, figsize= (10,6))

fig.subplots_adjust(hspace=0.8)
sns.countplot(credit_data['hours'],ax= axs[0], color= "violet")
axs[0].set_title("Distribution of Total Transactions",fontsize=15)
axs[0].set_facecolor("black")


sns.countplot(credit_data[(credit_data['Class']==0)]['hours'], ax= axs[1], color= 'gold')
axs[1].set_title("Distribution of Non-Fraudulent Transactions",fontsize=15)
axs[1].set_facecolor("black")

sns.countplot(credit_data[(credit_data['Class']==1)]['hours'], ax= axs[2], color= 'c')
axs[2].set_title("Distribution of Fraudulent Transactions",fontsize=15)
axs[2].set_facecolor("black")
#Scatter plot of Class vs Amount and Time for Normal Transactions

plt.figure(figsize=(20,8))

fig = plt.scatter(x=credit_data[credit_data['Class'] == 0]['Time'], y=credit_data[credit_data['Class'] == 0]['Amount'], color="dodgerblue", s=50, edgecolor='black')
plt.title("Time vs Transaction Amount in Normal Transactions", fontsize=20)
plt.xlabel("Time (in seconds)", fontsize=13)
plt.ylabel("Amount of Transaction", fontsize=13)

plt.show()
plt.figure(figsize=(20,8))

fig = plt.scatter(x=credit_data[credit_data['Class'] == 1]['Time'], y=credit_data[credit_data['Class'] == 1]['Amount'], color="c", s=100, edgecolor='black')
plt.title("Time vs Transaction Amount in Fraud Cases", fontsize=20)
plt.xlabel("Time (in seconds)", fontsize=13)
plt.ylabel("Amount of Transaction", fontsize=13)

plt.show()
import matplotlib.gridspec as gridspec
#Looking the V's features
columns = credit_data.iloc[:,1:29].columns

frauds = credit_data.Class == 1
normals = credit_data.Class == 0

grid = gridspec.GridSpec(14, 2)
plt.figure(figsize=(20,20*4))

for n, col in enumerate(credit_data[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(credit_data[col][frauds], color='darkorange', kde_kws={"color": "b", "lw": 1.5},  hist_kws=dict(alpha=1))
    sns.distplot(credit_data[col][normals],color='lightcoral', kde_kws={"color": "b", "lw": 1.5},  hist_kws=dict(alpha=1))
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(str(col), fontsize=20)
    ax.set_xlabel('')
plt.show()

q3= np.percentile(credit_data['Amount'], 75)
q1= np.percentile(credit_data['Amount'], 25)

# computing the interquartile range
IQR= q3-q1
# Find out lower and upper bound
lwr_bound = q1-(1.5*IQR)
upr_bound = q3+(1.5*IQR)
#creating a filter to remove values less than lower bound and greater than upper bound
filter_data= (credit_data['Amount'] < lwr_bound) | (credit_data['Amount'] > upr_bound)

# filtering data
outliers = credit_data[filter_data]['Amount']
fraud_outliers = credit_data[(credit_data['Class'] == 1) & filter_data]['Amount']
normal_outliers = credit_data[(credit_data['Class'] == 0) & filter_data]['Amount']

print(f"Total Number of Outliers : {outliers.count()}")
print(f"Number of Outliers in Fraudulent Class : {fraud_outliers.count()}")
print(f"No of Outliers in Normal Class : {normal_outliers.count()}")
print(f"Percentage of Fraud amount outliers : {round((fraud_outliers.count()/outliers.count())*100,2)}%")
# dropping the outliers

credit_data= credit_data.drop(outliers.index)
credit_data.reset_index(inplace=True, drop=True)
sns.distplot(credit_data['Amount'])
credit_data.drop('mins', axis= 1, inplace= True)
credit_data.drop('hours', axis= 1, inplace= True)

credit_data = pd.DataFrame(credit_data)
# saving the dataframe It is a clean data
credit_data.to_csv('credit_data.csv')
credit_data= pd.read_csv('C:/Users/Admin/PycharmProjects/Fruad_Transaction/credit_data.csv')
X= credit_data.drop('Class', axis= 1)
y= credit_data['Class']
#Feature EngineeringPowerTransformer
from sklearn.preprocessing import PowerTransformer
power = PowerTransformer(method='yeo-johnson', standardize=True)
df = power.fit_transform(X)
df= pd.DataFrame(data= df, columns= X.columns)
print(df.head())
print(df.shape)
fig, ax = plt.subplots(figsize=(15,8))
sns.distplot((df['Amount']), ax= ax, color = "g")
from imblearn.over_sampling import SMOTE
from collections import Counter
smt = SMOTE(random_state=2425, n_jobs=-1,sampling_strategy='auto', k_neighbors=5)
X_sm, y_sm = smt.fit_resample(df, y)
print('Resampled dataset shape {}'.format(Counter(y_sm)))
print('Before Resampled dataset shape {}'.format(Counter(y)))
sns.countplot(y_sm)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.30, random_state = 440)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)),
                     ('lr_classifier',LogisticRegression(random_state=0))])
pipeline_dt=Pipeline([('scalar2',StandardScaler()),
                     ('pca2',PCA(n_components=2)),
                     ('dt_classifier',DecisionTreeClassifier())])
pipeline_randomforest=Pipeline([('scalar3',StandardScaler()),
                     ('pca3',PCA(n_components=2)),
                     ('rf_classifier',RandomForestClassifier())])
pipeline_Knn=Pipeline([('scalar4',StandardScaler()),
                     ('pca4',PCA(n_components=2)),
                     ('rf_classifier',KNeighborsClassifier())])
pipeline_Gdb=Pipeline([('scalar5',StandardScaler()),
                     ('pca5',PCA(n_components=2)),
                     ('rf_classifier',GradientBoostingClassifier())])
pipeline_Adb=Pipeline([('scalar3',StandardScaler()),
                     ('pca3',PCA(n_components=2)),
                     ('rf_classifier',AdaBoostClassifier())])
## Lets make the list of pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_randomforest,pipeline_Knn, pipeline_Gdb,pipeline_Adb]
best_accuracy=0.0
best_classifier=0
best_pipeline=""
# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest',
             3: "KNN", 4: 'GradientBoosting ' , 5: 'AdaBoost'}

# Fit the pipelines
for pipe in pipelines:
	pipe.fit(x_train, y_train)
for i,model in enumerate(pipelines):
    print("{} Train Accuracy: {}".format(pipe_dict[i],model.score(x_train,y_train)))
for i,model in enumerate(pipelines):
    print("{} Train Accuracy: {}".format(pipe_dict[i],model.score(x_train,y_train)))
for i,model in enumerate(pipelines):
    if model.score(x_test,y_test)>best_accuracy:
        best_accuracy=model.score(x_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))
from sklearn.metrics import classification_report, f1_score

models2= []
models3= []
for i, model in enumerate(pipelines):
    pred = model.predict(x_test)
    Class_Report=classification_report(y_test, pred)
    models2.append(Class_Report)
    print("Model Name :",model,"\n",Class_Report)
    f1sc=f1_score(y_test, pred, average='weighted')
    models3.append(f1sc)
    print("Testing Accuracy of Data with Model",model,":",round(f1sc,2),"\n")
x= X = ["Logistic Regression" , "Decision Tree","RandomForest" ,
    "KNN " , "GradientBoosting ","AdaBoost"]
d2 = {"Accuracy of Testing Data" : models3}
data_frame2 = pd.DataFrame(d2,index=X)
data_frame2
cmodel = []
for i, model in enumerate(pipelines):
    pred = model.predict(x_test)
    cm = np.array(confusion_matrix(y_test, pred, labels=[1, 0]))
    cmodel.append(cm)

dc = {"Confusion Matrix": cmodel}
conf_mat = pd.DataFrame(dc, index=X)

conf_mat
plt.figure(figsize=(20,25))
plt.subplot(6,3,1)
sns.heatmap(cmodel[0],annot=True,fmt='d',cmap='Paired')
plt.title("Logistic Regression")
plt.subplot(6,3,2)
sns.heatmap(cmodel[1],annot=True,fmt='g',cmap='rocket')
plt.title("Decision Tree")
plt.subplot(6,3,3)
sns.heatmap(cmodel[2],annot=True,fmt='d',cmap='flare')
plt.title("RandomForest")
plt.subplot(6,3,4)
sns.heatmap(cmodel[3],annot=True,fmt='g',cmap='Set2')
plt.title("KNN")
plt.subplot(6,3,5)
sns.heatmap(cmodel[4],annot=True,fmt='d',cmap='PiYG')
plt.title("GradientBoosting")
plt.subplot(6,3,6)
sns.heatmap(cmodel[5],annot=True,fmt='g',cmap='Set3')
plt.title("AdaBoost")
plt.show()
from sklearn.model_selection import RepeatedStratifiedKFold

vmodel = []
acvalmodel = []

cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
for classifier, model in enumerate(pipelines):
    pred = model.predict(x_test)
    cross_val = cross_val_score(model, x_train, y_train, cv=cv, scoring='f1_weighted')
    acvalmodel.append(cross_val)
    print('Accuracy value :', cross_val)
    print('-----------------------------------')
    vmodel.append(round(cross_val.mean(), 2))
    print('Final Average Accuracy :', round(cross_val.mean(), 3))
    print("Final standard deviation :", cross_val.std())
    print('-----------------------------------')
dv={"RepeatedStratifiedKFold":acvalmodel,"Final Average Accuracy":vmodel}
crossval = pd.DataFrame(dv,index=X)
crossval
#cross_val['Final Average Accuracy'].plot(kind='bar')
crossval['Final Average Accuracy'].plot(kind='bar',figsize=(15,4),title='FinalAverage Accuray',colormap='Pastel1')
plt.show()
""""
#Hyperparameter_tuning
# Create a pipeline
pipe = make_pipeline((RandomForestClassifier()))
param_grid = {
    'randomforestclassifier': [RandomForestClassifier()],
    'randomforestclassifier__n_estimators': [10, 20, 50],
    'randomforestclassifier__max_features': ['auto', 'sqrt'],
    'randomforestclassifier__max_depth': [4, 7, 8],
    'randomforestclassifier__min_samples_leaf': [1, 10, 15, ]

}
# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
# Fit grid search
best_model = gridsearch.fit(x_train, y_train)
print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(x_test,y_test))
predictionRF = gridsearch.best_estimator_.predict(x_test)
print(confusion_matrix(y_test,predictionRF))
print("Accuracy Score {}".format(accuracy_score(y_test,predictionRF)))
print("Classification report: {}".format(classification_report(y_test,predictionRF)))"""

#Model
RF_model=Pipeline([('scalar3',StandardScaler()),
                     ('pca3',PCA()),
                     ('rf_classifier',RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=8, criterion='gini', min_samples_leaf= 15))])
RF_model.fit(x_train, y_train)
pred=RF_model.predict(x_test)
print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))
# Save the Modle to file in the current working directory
import pickle
Pkl_Filename = "Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(RF_model, file)
print("RF_model Saved")
#x_test[7895]
# Load the Model back from file
#Pkl_Filename = "Model.pkl"
#with open(Pkl_Filename, 'rb') as file:
#  Pickled_RF_model= pickle.load(file)

#Pickled_RF_model
# Use the Reloaded Model to
# Calculate the accuracy score and predict target values

# Calculate the Score
#score = Pickled_RF_model.score(x_test, y_test)
# Print the Score
#print("Test score: {0:.2f} %".format(100 * score))

# Predict the Labels using the reloaded Model
#Ypredict = Pickled_RF_model.predict(x_test)

#Ypredict
#predict the data using only one row

#features= np.array([-0.39953817, -0.97712131,  1.24544234,  0.94905287,  1.7934207 ,
#       -0.72105241,  0.71068923, -0.77640272,  1.36614126, -1.42028025,
 #       0.790728  ,  0.89664203, -0.02611889, -0.61222435,  1.12123451,
#        1.2181234 ,  0.33202657,  0.30275497,  0.838198  ,  0.55880226,
#        0.35222467,  0.51204671,  1.14737082, -0.18496812, -0.09257469,
#       -0.5097014 ,  0.79919997,  0.73473953,  0.42037289, -0.29209222])
#Ypredict = Pickled_RF_model.predict([features])

#Ypredict