#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[2]:


df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# In[3]:


test_df


# In[8]:


y = df['label']
X = df.drop(columns=['label'], axis=1)

X_train = X
y_train = y
# batch_size = int( 0.05 * X.shape[0])

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=int(batch_size * 0.8), test_size=int(batch_size * 0.2), random_state=42, stratify=y)


# In[9]:


from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def tune_hyperparameters(model, params, X_train, y_train):
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    return grid_search.best_estimator_


# In[10]:


#Bayes classifier
bayes_clf = GaussianNB()
bayes_params = {'var_smoothing': [1e-2, 1e-1, 1, 10]}
bayes_clf = tune_hyperparameters(bayes_clf, bayes_params, X_train.copy(), y_train.copy())


# In[ ]:


# Logistic Regression
logistic_clf = LogisticRegression(max_iter=3000)
logistic_params = {'C': [0.01, 0.01, 0.1, 1]}
logistic_clf = tune_hyperparameters(logistic_clf, logistic_params, X_train.copy(), y_train.copy())


# In[ ]:


# Tree Classifier
tree_clf = DecisionTreeClassifier()
tree_params = {'max_depth': [None, 2, 4, 6, 8, 16, 32]}
tree_clf = tune_hyperparameters(tree_clf, tree_params, X_train.copy(), y_train.copy())


# In[ ]:


# kNN Classifier
knn_clf = KNeighborsClassifier()
knn_params = {'n_neighbors': [2, 4, 6, 8, 16]}
knn_clf = tune_hyperparameters(knn_clf, knn_params, X_train.copy(), y_train.copy())


# In[ ]:


# Random Forest Classifier
forest_clf = RandomForestClassifier()
forest_params = {'n_estimators': [100, 150, 200, 250, 300], 'max_depth': [None, 2, 4, 8]}
forest_clf = tune_hyperparameters(forest_clf, forest_params, X_train.copy(), y_train.copy())


# In[ ]:


# SVM Classifier
svm_clf = SVC(probability=True)
svm_params = {'C': [0.1, 1, 10, 50, 100]}
svm_clf = tune_hyperparameters(svm_clf, svm_params, X_train.copy(), y_train.copy())


# In[ ]:


# Hard Voting Classifier
hard_voting_clf = VotingClassifier(estimators=[('bayes', bayes_clf), ('logistic', logistic_clf), ('tree', tree_clf), ('knn', knn_clf), ('forest', forest_clf), ('svm', svm_clf)], voting='hard')
# Soft Voting Classifier
soft_voting_clf = VotingClassifier(estimators=[('bayes', bayes_clf), ('logistic', logistic_clf), ('tree', tree_clf), ('knn', knn_clf), ('forest', forest_clf), ('svm', svm_clf)], voting='soft')


# In[ ]:


classifiers = {
    'Naive Bayes Classifier': bayes_clf,
    'Logistic Regression': logistic_clf,
    'Decision Tree Classifier': tree_clf,
    'kNN Classifier': knn_clf,
    'Random Forest Classifier': forest_clf,
    'SVM Classifier': svm_clf,
    'Hard Voting Classifier': hard_voting_clf,
    'Soft Voting Classifier': soft_voting_clf
}


# In[ ]:


@ignore_warnings(category=ConvergenceWarning)
def train_and_predict(clf, X, y):
    clf = classifiers[classifier]
    clf.fit(X, y)
    accuracy = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    return sum(accuracy) / len(accuracy)


for classifier in classifiers:
    accuracy = train_and_predict(classifier, X_train.copy(), y_train.copy())
    print(f'{classifier} Cross Validation Accuracy: {accuracy}')


# In[ ]:


def predictions(clf, batch_size, test_df):
    predictions_list = []
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i+batch_size, :]
        batch_predictions = clf.predict(batch)
        predictions_list.extend(batch_predictions)
    return predictions_list


# In[ ]:


bayes_predict = bayes_clf.predict(test_df)
logistic_predict = logistic_clf.predict(test_df)
knn_predict = knn_clf.predict(test_df)
tree_predict = tree_clf.predict(test_df)
forest_predict = forest_clf.predict(test_df)
svm_predict = svm_clf.predict(test_df)
hard_voting_predict = hard_voting_clf.predict(test_df)
soft_voting_predict = soft_voting_clf.predict(test_df)


# In[ ]:


final_data = {'ImageId': test_df.index + 1, 'Label': bayes_predict}
submission_bayes = pd.DataFrame(data=final_data)

final_data = {'ImageId': test_df.index + 1, 'Label': logistic_predict}
submission_logistic = pd.DataFrame(data=final_data)

final_data = {'ImageId': test_df.index + 1, 'Label': knn_predict}
submission_knn = pd.DataFrame(data=final_data)

final_data = {'ImageId': test_df.index + 1, 'Label': tree_predict}
submission_tree = pd.DataFrame(data=final_data)

final_data = {'ImageId': test_df.index + 1, 'Label': forest_predict}
submission_forest = pd.DataFrame(data=final_data)

final_data = {'ImageId': test_df.index + 1, 'Label': svm_predict}
submission_svm = pd.DataFrame(data=final_data)

final_data = {'ImageId': test_df.index + 1, 'Label': hard_voting_predict}
submission_hard_voting = pd.DataFrame(data=final_data)

final_data = {'ImageId': test_df.index + 1, 'Label': soft_voting_predict}
submission_soft_voting = pd.DataFrame(data=final_data)


# In[ ]:


submission_bayes.to_csv('submission_bayes.csv', index=False)
submission_logistic.to_csv('submission_logistic.csv', index=False)
submission_knn.to_csv('submission_knn.csv', index=False)
submission_tree.to_csv('submission_tree.csv', index=False)
submission_forest.to_csv('submission_forest.csv', index=False)
submission_svm.to_csv('submission_svm.csv', index=False)
submission_hard_voting.to_csv('submission_hard_voting.csv', index=False)
submission_soft_voting.to_csv('submission_soft_voting.csv', index=False)

