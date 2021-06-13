import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb

df=pd.read_csv("train.csv")
df



df.isnull().sum()


sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sb.set_style('whitegrid')
sb.countplot(x='Survived',data=df)

sb.countplot(x='Survived',hue='Sex',data=df)

sb.countplot(x='Survived',hue='Pclass',data=df)

sb.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=40)

sb.countplot(x='SibSp',data=df)

plt.figure(figsize=(12,7))
sb.boxplot(x='Pclass',y='Age',data=df)

def input_age(cols):
  Age=cols[0]
  Pclass=cols[1]
  if pd.isnull(Age):
    if Pclass==1:
      return 38
    elif Pclass==2:
      return 29
    else:
      return 24
  else:
    return Age


df['Age']=df[['Age','Pclass']].apply(input_age,axis=1)
df.head()


df.drop('Cabin',axis=1,inplace=True)
df.isnull().sum()

df.info()

sex=pd.get_dummies(df['Sex'],drop_first=True)
embark=pd.get_dummies(df['Embarked'],drop_first=True)

df.drop(['Sex','Embarked','Name','Ticket'],inplace=True,axis=1)
df

df=pd.concat([df,sex,embark],axis=1)
df

df.drop('Survived',axis=1)

df['Survived'].head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df.drop('Survived',axis=1),df['Survived'],test_size=0.25,random_state=10)

from sklearn.linear_model import LogisticRegression


lg=LogisticRegression()
lg.fit(x_train,y_train)

predictions=lg.predict(x_test)

from sklearn.metrics import confusion_matrix


accuracy=confusion_matrix(y_test,predictions)
accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)
accuracy