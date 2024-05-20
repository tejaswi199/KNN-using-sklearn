import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
df = pd.read_csv("/content/ClassifiedData (6).csv",index_col=0)
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
#print(scaled_features)
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
#print(df_feat)
X_train,X_test,y_train, y_test = train_test_split(scaled_features,
df['TARGET CLASS'], test_size=0.30)
#Initially with K=1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train,y_train)
pred1 = knn1.predict(X_test)
print("For K=1 results are:")
print(confusion_matrix(y_test,pred1))
print(classification_report(y_test,pred1))
# NOW WITH K=23
knn23 = KNeighborsClassifier(n_neighbors=23)
knn23.fit(X_train,y_train)
pred23 = knn23.predict(X_test)
print("For K=23 results are:")

print(classification_report(y_test,pred23))

