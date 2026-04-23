import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("ai_impact_student_performance_dataset.csv")

#Data Preprocessing:
data = df.copy()

#Fill NA values with the mean
data = data.fillna(data.median(numeric_only=True, skipna=True))

#Encoding
le = LabelEncoder()
le.fit(['Low', 'Medium', 'High'])
classification_y = le.transform(data['performance_category'])

#Dummies for gender column
data = pd.get_dummies(data, columns=['gender', 'grade_level', 'ai_tools_used', 'ai_usage_purpose'], drop_first=True, dtype=int)
x_reg = data.drop(columns=['student_id', 'final_score', 'passed', 'performance_category'])
y_reg = data['final_score']

x_clf = data.drop(columns=['student_id', 'final_score', 'passed', 'performance_category'])

#Feature selection using backward elimination
reg_selector = RFECV(estimator=RandomForestRegressor(random_state=42, n_jobs=-1), step=1, cv=5, scoring='r2')
reg_selector.fit(x_reg, y_reg)
optimized_x_reg = reg_selector.transform(x_reg)
print("Regression selected features: ", x_reg.columns[reg_selector.get_support()].tolist())

clf_selector = RFECV(estimator=RandomForestClassifier(random_state=42, n_jobs=-1), step=1, cv=5, scoring='accuracy')
clf_selector.fit(x_clf, classification_y)
scaler = StandardScaler()
optimized_x_clf = clf_selector.transform(x_clf)
optimized_x_clf = scaler.fit_transform(optimized_x_clf)
print('Classification selected features: ', x_clf.columns[clf_selector.get_support()])

#Save to files
np.save("optimized_x_reg.npy", optimized_x_reg)
np.save("optimized_x_clf.npy", optimized_x_clf)
np.save("y_reg.npy", y_reg.values)
np.save('classification_y.npy', classification_y)
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("Preprocessing done and saved")