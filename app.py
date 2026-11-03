import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Landsat Classification Dashboard",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Landsat Satellite Classification Dashboard")

st.write(
    """
    This application evaluates five trained machine learning classifiers
    on the UCI Statlog (Landsat Satellite) test dataset.

    Upload a test CSV containing Feature_1 to Feature_36 and the Class
    column, select a trained model, and view its evaluation results.
    """
)

st.info(
    "The application loads pre-trained models and does not retrain "
    "any model using the uploaded test dataset."
)

# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbor (KNN)": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

scaled_models = {
    "Logistic Regression",
    "K-Nearest Neighbor (KNN)",
    "Support Vector Machine"
}

# ---------------------------------------------------------
# LOAD PREPROCESSING ARTIFACT
# ---------------------------------------------------------

scaler = joblib.load("model/scaler.joblib")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Model Evaluation")

selected_model_name = st.sidebar.selectbox(
    "Select a Classification Model",
    list(model_files.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.write("Expected dataset structure:")
st.sidebar.write("• 36 input features")
st.sidebar.write("• 1 target column: Class")

# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------

if uploaded_file is None:

    st.subheader("Upload Test Data")

    st.write(
        "Upload `test_data.csv` using the sidebar to begin the model evaluation."
    )

else:

    data = pd.read_csv(uploaded_file)

    required_features = [f"Feature_{i}" for i in range(1, 37)]
    required_columns = required_features + ["Class"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        st.error(
            "The uploaded CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    else:

        st.success("Test dataset uploaded successfully.")

        st.subheader("Uploaded Dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric("Test Instances", data.shape[0])
        col2.metric("Input Features", len(required_features))
        col3.metric("Target Column", "Class")

        with st.expander("Preview Uploaded Test Data"):
            st.dataframe(data.head(10), use_container_width=True)

        # -------------------------------------------------
        # SEPARATE FEATURES AND TARGET
        # -------------------------------------------------

        X_test_upload = data[required_features]
        y_test_upload = data["Class"]

        # -------------------------------------------------
        # LOAD SELECTED TRAINED MODEL
        # -------------------------------------------------

        model = joblib.load(model_files[selected_model_name])

        # Apply scaling only to models trained on scaled data
        if selected_model_name in scaled_models:
            model_input = scaler.transform(X_test_upload)
        else:
            model_input = X_test_upload

        # -------------------------------------------------
        # PREDICTIONS
        # -------------------------------------------------

        predictions = model.predict(model_input)
        probabilities = model.predict_proba(model_input)

        # -------------------------------------------------
        # EVALUATION METRICS
        # -------------------------------------------------

        accuracy = accuracy_score(
            y_test_upload,
            predictions
        )

        auc = roc_auc_score(
            y_test_upload,
            probabilities,
            multi_class="ovr",
            average="weighted",
            labels=model.classes_
        )

        precision = precision_score(
            y_test_upload,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test_upload,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test_upload,
            predictions,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test_upload,
            predictions
        )

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        st.subheader(f"Evaluation Results: {selected_model_name}")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        metric_col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        metric_col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        metric_col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        metric_col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        metric_col6.metric(
            "MCC Score",
            f"{mcc:.4f}"
        )

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        st.subheader("Confusion Matrix")

        labels = model.classes_

        cm = confusion_matrix(
            y_test_upload,
            predictions,
            labels=labels
        )

        fig, ax = plt.subplots(figsize=(6, 4.5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )

        st.pyplot(fig, use_container_width=False)

        # -------------------------------------------------
        # CLASSIFICATION REPORT
        # -------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test_upload,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # -------------------------------------------------
        # PREDICTION PREVIEW
        # -------------------------------------------------

        st.subheader("Prediction Preview")

        prediction_preview = pd.DataFrame({
            "Actual Class": y_test_upload.values,
            "Predicted Class": predictions
        })

        st.dataframe(
            prediction_preview.head(20),
            use_container_width=True
        )

        correct_predictions = np.sum(
            y_test_upload.values == predictions
        )

        st.caption(
            f"Correctly classified {correct_predictions} "
            f"out of {len(predictions)} uploaded test instances."
        )