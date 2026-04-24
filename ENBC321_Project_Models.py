import pandas as pd
import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

optimized_x_reg = np.load('optimized_x_reg.npy')
optimized_x_clf = np.load('optimized_x_clf.npy')
y_reg = np.load('y_reg.npy')
classification_y = np.load('classification_y.npy')
le = pickle.load(open('label_encoder.pkl', 'rb'))

#Machine Learning Techniques (Supervised):

#Linear Regression
linear_Reg = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(optimized_x_reg, y_reg, test_size = 0.2, random_state=10)
linear_Model = linear_Reg.fit(x_train, y_train)
y_pred_linear = linear_Model.predict(x_test)

plt.scatter(y_test, y_pred_linear, color='blue', alpha=0.6, label='Predictions')

min_val = min(y_test.min(), y_pred_linear.min())
max_val = max(y_test.max(), y_pred_linear.max())
plt.plot([min_val, max_val], [min_val, max_val], color='black', alpha=0.6, label='Predictions')
plt.title('Linear Regression: Actual Vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

#Logistic Regression
class_x_train, class_x_test, class_y_train, class_y_test = train_test_split(optimized_x_clf, classification_y, test_size=0.2, random_state=10)
log_Reg = LogisticRegression(max_iter=5000)
log_Model = log_Reg.fit(class_x_train, class_y_train)
y_pred_log = log_Reg.predict(class_x_test)

ConfusionMatrixDisplay.from_estimator(log_Model, class_x_test, class_y_test, display_labels=le.classes_)
plt.title("Logistic Regression - Confusion Matrix")
plt.show()
#Support Vector Machines (SVM)
svm_model = SVC(kernel='linear', C=1.0)


#Decision Trees

#Random Forests

#k-Nearest Neighbors (k-NN)

#Naive Bayes

#Machine Learning Techniques (Unsupervised):

#k-Means Clustering

#Hierarchical Clustering

#Prinicpal Component Analysis (PCA)

#Deep Learning:
