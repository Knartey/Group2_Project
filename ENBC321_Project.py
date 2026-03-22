import pandas as pd

df = pd.read_csv("ai_impact_student_performance_dataset.csv")

#Data Preprocessing:
copy = df.copy()

#Fill NA values with the mean
copy = copy.fillna(df.median(numeric_only=True, skipna=True))

#Machine Learning Techniques (Supervised):

#Linear Regression

#Logistic Regression

#Support Vector Machines (SVM)

#Decision Trees

#Random Forests

#k-Nearest Neighbors (k-NN)

#Naive Bayes

#Machine Learning Techniques (Unsupervised):

#k-Means Clustering

#Hierarchical Clustering

#Prinicpal Component Analysis (PCA)

#Deep Learning:
