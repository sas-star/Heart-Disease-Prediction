import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.colors as mcolors
import matplotlib.cm as cm
sns.set_theme(
    style="whitegrid",
    palette="coolwarm",
    font_scale=1.2
)

plt.rcParams["figure.facecolor"] = "#ffffff"
plt.rcParams["axes.facecolor"] = "#ffffff"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.linestyle"] = "--"
# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Heart Disease Prediction ",
    page_icon="❤️",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("Cleaned_heart_disease.csv")

# Load trained model
model = joblib.load("xgboost.pkl")
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prob" not in st.session_state:
    st.session_state.prob = None
# ==========================
# TITLE
# ==========================

st.title("❤️ Heart Disease Prediction ")
st.markdown("---")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("Patient Information")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=50
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

bp = st.sidebar.number_input(
    "Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)

cholesterol = st.sidebar.number_input(
    "Cholesterol",
    min_value=100,
    max_value=500,
    value=200
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=25.0
)

sleep = st.sidebar.number_input(
    "Sleep Hours",
    min_value=1.0,
    max_value=12.0,
    value=7.0
)

exercise = st.sidebar.selectbox(
    "Exercise Habits",
    ["Low", "Medium", "High"]
)

smoking = st.sidebar.selectbox(
    "Smoking",
    ["Yes", "No"]
)

diabetes = st.sidebar.selectbox(
    "Diabetes",
    ["Yes", "No"]
)

sugar = st.sidebar.selectbox(
    "Sugar Consumption",
    ["Low", "Medium", "High"]
)

# ==========================
# ENCODING
# ==========================

gender_val = 1 if gender == "Male" else 0

smoking_val = 1 if smoking == "Yes" else 0

diabetes_val = 1 if diabetes == "Yes" else 0

exercise_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

sugar_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}
# ======================================
# TABS
# ======================================

tab1, tab2, tab3 = st.tabs(
    [
        "🩺 Prediction",
        "🏆 Model Performance",
        "⭐ Feature Importance"
    ]
)

# ======================================
# TAB 1
# ======================================
with tab1:

    st.subheader("Heart Disease Prediction")

    if "predicted" not in st.session_state:
        st.session_state.predicted = False

    col1, col2 = st.columns(2)

    with col1:
        predict_btn = st.button(
            "🔍 Predict Heart Disease Risk",
            key="predict_btn"
        )

    with col2:
        reset_btn = st.button(
            "🔄 Reset",
            key="reset_btn"
        )

    if reset_btn:
        st.session_state.predicted = False
        st.session_state.prediction = None
        st.session_state.prob = None
        st.rerun()

    if predict_btn:

        input_data = np.array([[
            age,
            gender_val,
            bp,
            cholesterol,
            exercise_map[exercise],
            smoking_val,
            0,
            diabetes_val,
            bmi,
            0,
            0,
            0,
            0,
            1,
            sleep,
            sugar_map[sugar],
            150,
            100,
            5,
            10
        ]])

        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)[0]

        st.session_state.prediction = prediction
        st.session_state.prob = prob
        st.session_state.predicted = True

    if st.session_state.predicted:

        prediction = st.session_state.prediction
        prob = st.session_state.prob

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Disease Probability",
            f"{prob[1]*100:.2f}%"
        )

        c2.metric(
            "Model",
            "XGBoost"
        )

        c3.metric(
            "Dataset Rows",
            len(df)
        )

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.write("Prediction:", prediction[0])

        if hasattr(model, "predict_proba"):
         st.write(
        "Disease Probability:",
        round(prob[1] * 100, 2),
        "%"
    )
        # ======================
        # Probability Chart
        # ======================

        st.subheader("Prediction Probability")

        prob_df = pd.DataFrame({
            "Risk": ["Low Risk", "High Risk"],
            "Probability": [prob[0], prob[1]]
        })

        st.bar_chart(
            prob_df.set_index("Risk")
        )

        # ======================
        # Histogram
        # ======================

        st.subheader("📊 Age Distribution")

        fig, ax = plt.subplots(figsize=(12,6))

        cmap = cm.get_cmap("RdYlGn_r")

        n, bins, patches = ax.hist(
            df["Age"],
            bins=20,
            edgecolor="black"
        )

        for patch, left in zip(patches, bins):

            normalized = (
                left - df["Age"].min()
            ) / (
                df["Age"].max() - df["Age"].min()
            )

            patch.set_facecolor(
                cmap(normalized)
            )

        norm = mcolors.Normalize(
            vmin=df["Age"].min(),
            vmax=df["Age"].max()
        )

        sm = cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )

        sm.set_array([])

        cbar = plt.colorbar(
            sm,
            ax=ax
        )

        cbar.set_label(
            "Age Risk Scale"
        )

        ax.set_title(
            "Age Distribution",
            fontsize=18,
            fontweight="bold"
        )

        ax.grid(
            linestyle="--",
            alpha=0.5
        )

        st.pyplot(fig)

        # ======================
        # Box Plot
        # ======================

        st.subheader("📦 Outlier Analysis")

        fig, ax = plt.subplots(figsize=(10,5))

        sns.boxplot(
            data=df[
                [
                    "Age",
                    "BMI",
                    "Cholesterol Level"
                ]
            ],
            palette="coolwarm",
            ax=ax
        )

        ax.grid(
            linestyle="--",
            alpha=0.5
        )

        st.pyplot(fig)

        # ======================
        # Heatmap
        # ======================

        st.subheader("🔥 Correlation Heatmap")

        corr = df.select_dtypes(
            include=["int64", "float64"]
        ).corr()

        fig, ax = plt.subplots(
            figsize=(12,8)
        )

        sns.heatmap(
            corr,
            cmap="coolwarm",
            linewidths=0.5,
            square=True,
            ax=ax
        )

        st.pyplot(fig)
# ======================================
# TAB 2
# ======================================

with tab2:

    st.subheader("🏆 Model Accuracy Comparison")

    model_df = pd.DataFrame({

        "Model":[
            "Decision Tree",
            "KNN",
            "Logistic",
            "Naive Bayes",
            "Random Forest",
            "SVM",
            "XGBoost"
        ],

        "Accuracy":[
            0.83,
            0.80,
            0.85,
            0.82,
            0.89,
            0.87,
            0.91
        ]
    })

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    sns.barplot(
        data=model_df,
        x="Model",
        y="Accuracy",
        palette="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Machine Learning Model Comparison",
        fontsize=18,
        fontweight="bold"
    )

    ax.grid(
        linestyle="--",
        alpha=0.5
    )

    plt.xticks(rotation=30)

    st.pyplot(fig)

# ======================================
# TAB 3
# ======================================

with tab3:

    if hasattr(model, "feature_importances_"):

        feature_names = [
            "Age",
            "Gender",
            "Blood Pressure",
            "Cholesterol",
            "Exercise",
            "Smoking",
            "Family History",
            "Diabetes",
            "BMI",
            "High BP",
            "Low HDL",
            "High LDL",
            "Alcohol",
            "Stress",
            "Sleep",
            "Sugar",
            "Triglyceride",
            "Fasting Sugar",
            "CRP",
            "Homocysteine"
        ]

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        imp_df = imp_df.sort_values(
            "Importance",
            ascending=False
        )

        st.subheader("⭐ Feature Importance")

        fig, ax = plt.subplots(
            figsize=(10,6)
        )

        sns.barplot(
            data=imp_df.head(10),
            x="Importance",
            y="Feature",
            palette="coolwarm",
            ax=ax
        )

        ax.grid(
            linestyle="--",
            alpha=0.5
        )

        st.pyplot(fig)
