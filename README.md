# Landsat Satellite Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for classifying satellite image pixels into different land-cover classes.

The project uses the Statlog (Landsat Satellite) dataset from the UCI Machine Learning Repository. Five classification algorithms are trained on the same dataset and evaluated on an independent test dataset using multiple performance metrics.

The implemented classification models are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine (SVM)

The trained models are integrated into an interactive Streamlit application where a user can upload test data, select a trained model, and view its evaluation results without retraining the model on the uploaded data.

## b. Dataset Description

**Dataset Name:** Statlog (Landsat Satellite)

**Source:** UCI Machine Learning Repository

**Task:** Multiclass Classification

**Total Instances:** 6,435

**Number of Input Features:** 36

**Training Instances:** 4,435

**Test Instances:** 2,000

**Target Column:** Class

The dataset contains multispectral satellite image information. Each observation represents the multispectral values of pixels from a 3 x 3 neighbourhood in a satellite image. The objective is to predict the land-cover classification associated with the central pixel.

The dataset contains six observed class labels:

- Class 1
- Class 2
- Class 3
- Class 4
- Class 5
- Class 7

The UCI dataset does not contain observations for Class 6.

The original training and test datasets provided by UCI are used in this project. The trained models are evaluated using the provided test dataset.

## c. GitHub Repository Link

**GitHub Repository:**

[GitHub Repository - Landsat ML Classification](https://github.com/mkemuelluther/landsat-ml-classification)

## d. Models Used and Evaluation Results

All five machine learning models were trained using the same Landsat training dataset and evaluated using the same independent test dataset.

For scale-sensitive algorithms such as Logistic Regression, KNN, and Support Vector Machine, StandardScaler was fitted only using the training data and then applied to the test data.

The following evaluation metrics were calculated:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

<table>
<tr>
<th>ML Model Name</th>
<th>Accuracy</th>
<th>AUC</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
<th>MCC</th>
</tr>

<tr>
<td>Logistic Regression</td>
<td>0.8395</td>
<td>0.9789</td>
<td>0.8283</td>
<td>0.8395</td>
<td>0.8296</td>
<td>0.8032</td>
</tr>

<tr>
<td>Decision Tree</td>
<td>0.8490</td>
<td>0.9089</td>
<td>0.8505</td>
<td>0.8490</td>
<td>0.8497</td>
<td>0.8146</td>
</tr>

<tr>
<td>K-Nearest Neighbor (KNN)</td>
<td>0.9045</td>
<td>0.9817</td>
<td>0.9035</td>
<td>0.9045</td>
<td>0.9037</td>
<td>0.8827</td>
</tr>

<tr>
<td>Gaussian Naive Bayes</td>
<td>0.7965</td>
<td>0.9588</td>
<td>0.8165</td>
<td>0.7965</td>
<td>0.8036</td>
<td>0.7531</td>
</tr>

<tr>
<td>Support Vector Machine</td>
<td>0.8955</td>
<td>0.9807</td>
<td>0.8923</td>
<td>0.8955</td>
<td>0.8925</td>
<td>0.8718</td>
</tr>
</table>

## Model Performance Observations

<table>
<tr>
<th>ML Model Name</th>
<th>Observation about Model Performance</th>
</tr>

<tr>
<td>Logistic Regression</td>
<td>Logistic Regression achieved an accuracy of 0.8395 and a high AUC of 0.9789. The strong AUC indicates good overall class discrimination, although its F1 score and MCC were lower than those of KNN and SVM.</td>
</tr>

<tr>
<td>Decision Tree</td>
<td>Decision Tree achieved an accuracy of 0.8490 and an F1 score of 0.8497. It performed better than Logistic Regression on accuracy, precision, recall, F1 and MCC, but produced the lowest AUC among the five models.</td>
</tr>

<tr>
<td>K-Nearest Neighbor (KNN)</td>
<td>KNN produced the strongest overall results with an accuracy of 0.9045, AUC of 0.9817, F1 score of 0.9037 and MCC of 0.8827. Standardization of the input features helped the distance-based classifier operate on comparable feature scales.</td>
</tr>

<tr>
<td>Gaussian Naive Bayes</td>
<td>Gaussian Naive Bayes achieved an accuracy of 0.7965. Although its AUC remained relatively high at 0.9588, its accuracy, recall and MCC were the lowest among the evaluated models, indicating comparatively weaker final class predictions on this dataset.</td>
</tr>

<tr>
<td>Support Vector Machine</td>
<td>SVM achieved an accuracy of 0.8955 and an AUC of 0.9807. Its performance was close to KNN and substantially stronger than Logistic Regression, Decision Tree and Naive Bayes on most classification metrics.</td>
</tr>
</table>

## Overall Winner for the Dataset

**K-Nearest Neighbor (KNN)** was the best-performing model for this dataset.

KNN achieved:

Accuracy: **0.9045**
AUC: **0.9817**
Precision: **0.9035**
Recall: **0.9045**
F1 Score: **0.9037**
MCC Score: **0.8827**

KNN obtained the highest value among the five evaluated classifiers for all six reported evaluation metrics. Therefore, KNN was selected as the overall winner for the Landsat Satellite dataset.

## Streamlit Application

An interactive Streamlit web application was developed to evaluate the saved machine learning models.

The application provides:

CSV test dataset upload
Model selection dropdown
Evaluation metrics
Confusion matrix
Classification report
Prediction preview

The application loads previously trained models from saved model files. It does not retrain the models using the uploaded test dataset.

## Project Structure

<pre>
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── Landsat_ML_Assignment_2.ipynb
├── model_comparison_results.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── scaler.joblib
</pre>

## Running the Streamlit Application

Install the required Python packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

Upload test_data.csv through the application and select any of the five trained classifiers to view the corresponding evaluation results.