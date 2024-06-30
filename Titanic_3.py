#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kazirafi17/TitanicSurvivalPrediction/blob/main/Titanic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# #### Load Data

# In[2]:


# Import necessary libraries

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[3]:


# Load Titanic dataset
df = pd.read_csv('titanic.csv')

# Display columns to inspect their names and content
df.head(3)


# In[4]:


df.info()


# In[5]:


df.describe().T


# In[6]:


df.isnull().sum()


# #### Plotting

# In[7]:


# Bar plot of survival by Pclass, Sex, Embarked
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

sns.countplot(data=df, x = 'Pclass', hue = 'Survived', ax=axes[0])
sns.countplot(data=df, x = 'Sex', hue = 'Survived', ax=axes[1])


# In[8]:


# Histogram of ages

plt.figure(figsize=(10, 5))
df['Age'].hist(bins=20)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[9]:


# Histogram of fare

plt.figure(figsize=(10, 5))
df['Fare'].hist(bins=20)
plt.title('Fare Distribution of Passengers')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()


# In[10]:


# Increase the plot size
plt.figure(figsize=(9, 6))

# Create the boxplot
sns.boxplot(x='Pclass', y='Age', data=df)


# In[11]:


# Set the figure size
plt.figure(figsize=(10, 6))

# Create the boxplot
sns.boxplot(x='Pclass', y='Fare', data=df)

# Show the plot
plt.show()


# In[12]:


# Create the countplot
sns.countplot(x='Survived', hue='Pclass', data=df)

# Show the plot
plt.show()


# In[13]:


plt.figure(figsize=(15,5))
sns.distplot(df['Fare'],bins=40)


# In[14]:


df_fare = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
plt.figure(figsize=(15,5))
sns.distplot(df_fare,bins=40)


# #### EDA

# In[15]:


fare= df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df['Fare'] = fare


# In[16]:


#df['Sex'] = df['Sex'].astype(str)
df.drop(['Name'],inplace=True,axis=1)


# In[17]:


df = pd.get_dummies(df,columns=['Sex'],drop_first=True)


# In[18]:


X = df.drop(['Survived'],axis=1)
y = df['Survived']


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Model Selection

# In[20]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def evaluate_classifiers(data, target_col):


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # List of classifiers to evaluate
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }

    # Dictionary to store the results
    results = {}

    # Evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None

        # Store the results
        results[name] = {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'ROC AUC': roc_auc
        }

    return pd.DataFrame(results).T



# Assume 'target' is the name of the column to predict
results = evaluate_classifiers(df, 'target')

print(results)


# Based on this data, XGBoost is the best choice. Now, the next step is corss validation and hyperparameter tuning.

# Hyperparameter tuning

# In[21]:


from sklearn.model_selection import cross_val_score

# Initialize and train the model
model = XGBClassifier(colsample_bytree=1.0, gamma=0.3, learning_rate=0.1, max_depth=4, n_estimators=200, subsample=0.8)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10)
print(f'Mean cross-validation score: {cv_scores.mean()}')


# In[22]:


import pickle

pickle.dump(model,open('titanic.pkl','wb'))


# In[23]:


pickled_model=pickle.load(open('titanic.pkl','rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




