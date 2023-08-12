#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC


# In[2]:


df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
df['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NAN


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df_numeric = df[['Age', 'SibSp', 'Parch', 'Fare']]
df_categoricals = df[['Survived', 'Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]
df.describe().columns


# In[6]:


for column in df_numeric:
    plt.hist(df_numeric[column])
    plt.title(column)
    plt.show()


# In[7]:


print(df_numeric.corr())
sns.heatmap(df_numeric.corr(), cmap='gray')


# In[8]:


pd.pivot_table(df, index='Survived', values=['Age', 'SibSp', 'Parch', 'Fare'])


# In[9]:


for column in df_categoricals.columns:
    value_counts = df_categoricals[column].value_counts().sort_index()
    sns.barplot(x=value_counts.index, y=value_counts)
    plt.title(column)
    plt.show()


# In[10]:


print(pd.pivot_table(df, index='Survived', columns='Pclass', values='Ticket', aggfunc='count'))
print(pd.pivot_table(df, index='Survived', columns='Sex', values='Ticket', aggfunc='count'))
print(pd.pivot_table(df, index='Survived', columns='Embarked', values='Ticket', aggfunc='count'))


# In[11]:


df['cabin_multiple'] = df.Cabin.apply(lambda x : 0 if pd.isna(x) else len(x.split(' ')))
df['cabin_multiple'].value_counts()


# In[12]:


pd.pivot_table(df, index='Survived', columns='cabin_multiple', values='Ticket', aggfunc='count')


# In[13]:


df['cabin_adv'] = df.Cabin.apply(lambda x: str(x)[0])
print(df.cabin_adv.value_counts())
pd.pivot_table(df, index='Survived', columns='cabin_adv', values='Name', aggfunc='count')


# In[14]:


df['numeric_ticket'] = df.Ticket.apply(lambda x : 1 if x.isnumeric() else 0)
df['ticket_letters'] = df.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 0)
print(df.numeric_ticket.value_counts())
# pd.set_option("max_rows", None)
df.ticket_letters.value_counts()


# In[15]:


pd.pivot_table(df, index='Survived', columns='numeric_ticket', values='Ticket', aggfunc='count')


# In[16]:


pd.pivot_table(df, index='Survived', columns='ticket_letters', values='Ticket', aggfunc='count')


# In[17]:


df.Name.head()
df['name_title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
df['name_title'].value_counts()


# In[18]:


pd.pivot_table(df, index='Survived', columns='name_title', values='Ticket', aggfunc='count')


# In[19]:


all_data = pd.concat([df, test])


# In[20]:


all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = df.Cabin.apply(lambda x: str(x)[0])
all_data['numeric_ticket'] = df.Ticket.apply(lambda x : 1 if x.isnumeric() else 0)
all_data['ticket_letters'] = df.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 0)
all_data['name_title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

all_data.Age = all_data.Age.fillna(df.Age.mean())
all_data.Fare = all_data.Fare.fillna(df.Fare.mean())
all_data.dropna(subset=['Embarked'], inplace=True)


# In[21]:


all_data['norm_fare'] = np.log(all_data.Fare + 1)
all_data['norm_fare'].hist()


# In[22]:


all_data.Pclass = all_data.Pclass.astype(str)
all_dummies = pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'norm_fare', 'Embarked', 'cabin_adv', 'cabin_multiple', 'numeric_ticket', 'name_title', 'train_test']])
X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis=1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis=1)
y_train = all_data[all_data.train_test == 1].Survived
y_train.shape


# In[23]:


scaler = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']] = scaler.fit_transform(all_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']])
X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis=1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis=1)
y_train = all_data[all_data.train_test == 1].Survived


# In[24]:


# Naive Bayes
bayes_clf = GaussianNB()
cross_val = cross_val_score(bayes_clf, X_train_scaled, y_train, cv=5)
print(cross_val.mean())


# In[25]:


# Logistic Regression
lr_clf = LogisticRegression()
cross_val = cross_val_score(lr_clf, X_train_scaled, y_train, cv=5)
print(cross_val.mean())


# In[26]:


# Tree
tree_clf = tree.DecisionTreeClassifier(max_depth=4)
cross_val = cross_val_score(tree_clf, X_train_scaled, y_train, cv=5)
print(cross_val.mean())


# In[27]:


# SVM
svm_clf = SVC(C=1)
cross_val = cross_val_score(svm_clf, X_train_scaled, y_train, cv=5)
print(cross_val.mean())


# In[28]:


# kNN
knn_clf = KNeighborsClassifier(n_neighbors=10)
cross_val = cross_val_score(knn_clf, X_train_scaled, y_train, cv=5)
print(cross_val.mean())


# In[29]:


# Random Forest
forest_clf = RandomForestClassifier(n_estimators=9, n_jobs=-1)
cross_val = cross_val_score(forest_clf, X_train_scaled, y_train, cv=5)
print(cross_val.mean())


# In[30]:


# Voting Classifier
voting_clf = VotingClassifier(estimators=[('forest_clf', forest_clf), ('svm_clf', svm_clf), ('bayes_clf', bayes_clf), ('knn_clf', knn_clf), ('tree_clf', tree_clf), ('lr_clf', lr_clf)], voting='hard')
cross_val = cross_val_score(voting_clf, X_train_scaled, y_train, cv=5)
print(cross_val.mean())


# In[31]:


bayes_clf.fit(X_train_scaled, y_train)
lr_clf.fit(X_train_scaled, y_train)
knn_clf.fit(X_train_scaled, y_train)
tree_clf.fit(X_train_scaled, y_train)
forest_clf.fit(X_train_scaled, y_train)
svm_clf.fit(X_train_scaled, y_train)
voting_clf.fit(X_train_scaled, y_train)

bayes_predict = bayes_clf.predict(X_test_scaled)
lr_predict = lr_clf.predict(X_test_scaled)
knn_predict = knn_clf.predict(X_test_scaled)
tree_predict = tree_clf.predict(X_test_scaled)
forest_predict = forest_clf.predict(X_test_scaled)
svm_predict = svm_clf.predict(X_test_scaled)
voting_predict = voting_clf.predict(X_test_scaled)


# In[32]:


final_data = {'PassengerId': test.PassengerId, 'Survived': bayes_predict}
submission_bayes = pd.DataFrame(data=final_data)

final_data = {'PassengerId': test.PassengerId, 'Survived': lr_predict}
submission_lr = pd.DataFrame(data=final_data)

final_data = {'PassengerId': test.PassengerId, 'Survived': knn_predict}
submission_knn = pd.DataFrame(data=final_data)

final_data = {'PassengerId': test.PassengerId, 'Survived': tree_predict}
submission_tree = pd.DataFrame(data=final_data)

final_data = {'PassengerId': test.PassengerId, 'Survived': forest_predict}
submission_forest = pd.DataFrame(data=final_data)

final_data = {'PassengerId': test.PassengerId, 'Survived': svm_predict}
submission_svm = pd.DataFrame(data=final_data)

final_data = {'PassengerId': test.PassengerId, 'Survived': voting_predict}
submission_voting = pd.DataFrame(data=final_data)


# In[33]:


for submission in [submission_bayes, submission_lr, submission_knn, submission_tree, submission_forest, submission_svm, submission_voting]:
    submission.Survived = submission.Survived.astype(int)


# In[34]:


submission_bayes.to_csv('submission_bayes.csv', index=False)
submission_lr.to_csv('submission_lr.csv', index=False)
submission_knn.to_csv('submission_knn.csv', index=False)
submission_tree.to_csv('submission_tree.csv', index=False)
submission_forest.to_csv('submission_forest.csv', index=False)
submission_svm.to_csv('submission_svm.csv', index=False)
submission_voting.to_csv('submission_voting.csv', index=False)

