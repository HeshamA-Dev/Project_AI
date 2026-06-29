import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="AI Job Impact Predictor",
    page_icon="✌🏼",
    layout="wide"
)


# --------------------------------------------------
# Load data and model
# --------------------------------------------------

DATA_PATH = "data/cleaned/ai_job_market_insights_cleaned.csv"
MODEL_PATH = "model/job_growth_model.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset",
        "Data Cleaning",
        "Visualizations",
        "Machine Learning",
        "Live Prediction"
    ]
)


# --------------------------------------------------
# Home page
# --------------------------------------------------

if page == "Home":
    st.title(" AI Job Impact Predictor")

    st.write("""
    This project analyzes job market data and predicts whether a job is likely to grow in the future.
    The project includes data cleaning, dataset optimization, visualizations, machine learning, and a live Streamlit demo.
    """)

    st.subheader("Project Goal")

    st.write("""
    The goal is to build a machine learning model that predicts whether a job is likely to grow or not.
    The model predicts one of two classes:
    """)

    st.write("""
    - Growing
    - Not Growing
    """)

    st.subheader("Project Workflow")

    st.write("""
    1. Load and understand the dataset  
    2. Clean and prepare the data  
    3. Visualize important patterns  
    4. Train a scikit-learn classification model  
    5. Use the model in a Streamlit app for live predictions  
    """)


# --------------------------------------------------
# Dataset page
# --------------------------------------------------

elif page == "Dataset":
    st.title("Dataset Overview")

    st.write("""
    The dataset contains job market information related to artificial intelligence, automation risk,
    required skills, salary, remote work, and job growth.
    """)

    st.subheader("Dataset Shape")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Columns")
    st.write(list(df.columns))

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().to_frame(name="Missing Values"))


# --------------------------------------------------
# Data Cleaning page
# --------------------------------------------------

elif page == "Data Cleaning":
    st.title("Data Cleaning Pipeline")

    st.write("""
    The data cleaning process prepared the dataset for analysis and machine learning.
    """)

    st.subheader("Cleaning Steps")

    st.write("""
    - Checked missing values
    - Checked duplicate rows
    - Cleaned categorical text values
    - Converted salary values from float to integer
    - Converted Remote_Friendly from Yes/No to 1/0
    - Checked possible salary outliers
    - Saved the cleaned dataset as a new CSV file
    """)

    st.subheader("Optimizations")

    st.write("""
    Salary values were rounded and converted to integers because salary values do not need decimal places
    for this project.
    """)

    st.write("""
    Remote_Friendly was converted into a binary variable:
    """)

    st.write("""
    - Yes = 1
    - No = 0
    """)

    st.subheader("Remote-Friendly Percentage")

    remote_table = df["Remote_Friendly"].map({
        1: "Yes",
        0: "No"
    }).value_counts(normalize=True) * 100

    remote_table = remote_table.round(1).to_frame(name="Percentage (%)")

    st.dataframe(remote_table)


# --------------------------------------------------
# Visualizations page
# --------------------------------------------------

elif page == "Visualizations":
    st.title("Visualizations")

    st.subheader("Salary Distribution")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["Salary_USD"], bins=20, edgecolor="black")
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Salary in USD")
    ax.set_ylabel("Number of Jobs")

    st.pyplot(fig)

    st.subheader("Automation Risk Distribution")

    risk_counts = df["Automation_Risk"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    risk_counts.plot(kind="bar", ax=ax2)
    ax2.set_title("Automation Risk Distribution")
    ax2.set_xlabel("Automation Risk")
    ax2.set_ylabel("Number of Jobs")

    st.pyplot(fig2)

    st.subheader("Average Salary by Industry")

    salary_by_industry = df.groupby("Industry")["Salary_USD"].mean().sort_values()

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    salary_by_industry.plot(kind="barh", ax=ax3)
    ax3.set_title("Average Salary by Industry")
    ax3.set_xlabel("Average Salary in USD")
    ax3.set_ylabel("Industry")

    st.pyplot(fig3)

    st.subheader("Job Growth Distribution")

    growth_counts = df["Job_Growth_Projection"].value_counts()

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    growth_counts.plot(kind="bar", ax=ax4)
    ax4.set_title("Job Growth Projection Distribution")
    ax4.set_xlabel("Job Growth Projection")
    ax4.set_ylabel("Number of Jobs")

    st.pyplot(fig4)


# --------------------------------------------------
# Machine Learning page
# --------------------------------------------------

elif page == "Machine Learning":
    st.title("Machine Learning Model")

    st.write("""
    The machine learning task is a classification problem.
    The goal is to predict whether a job is likely to grow or not.
    """)

    st.subheader("Target Variable")

    st.write("""
    The original target variable is:
    """)

    st.code("Job_Growth_Projection")

    st.write("""
    For the model, the target was converted into a binary variable:
    """)

    st.write("""
    - Growth = Growing
    - Stable = Not Growing
    - Decline = Not Growing
    """)

    st.subheader("Model Features")

    st.write("""
    The model uses these input features:
    """)

    st.write("""
    - Job_Title
    - Company_Size
    - Location
    - AI_Adoption_Level
    - Automation_Risk
    - Remote_Friendly
    """)

    st.subheader("Removed Features")

    st.write("""
    Some columns were not used as model inputs:
    """)

    st.write("""
    - Job_Growth_Projection was removed because it is the target variable.
    - Salary_USD was removed because exact salary is difficult to know for a general prediction.
    - Required_Skills was removed because skills can vary strongly inside the same job title.
    - Industry was removed because similar jobs can exist across many different industries.
    """)

    st.subheader("Model")

    st.write("""
    A RandomForestClassifier from scikit-learn was used.
    Categorical columns were encoded using OneHotEncoder.
    The preprocessing and model were combined in a scikit-learn Pipeline.
    """)


# --------------------------------------------------
# Live Prediction page
# --------------------------------------------------

elif page == "Live Prediction":
    st.title("Live Prediction")

    st.write("""
    Enter general job information below and let the model predict whether this job is likely to grow or not.
    """)

    col1, col2 = st.columns(2)

    with col1:
        job_title = st.selectbox("Job Title", sorted(df["Job_Title"].unique()))
        company_size = st.selectbox("Company Size", sorted(df["Company_Size"].unique()))
        location = st.selectbox("Location", sorted(df["Location"].unique()))

    with col2:
        ai_adoption_level = st.selectbox("AI Adoption Level", sorted(df["AI_Adoption_Level"].unique()))
        automation_risk = st.selectbox("Automation Risk", sorted(df["Automation_Risk"].unique()))

        remote_friendly_label = st.selectbox("Remote Friendly", ["Yes", "No"])
        remote_friendly = 1 if remote_friendly_label == "Yes" else 0

    input_data = pd.DataFrame([{
        "Job_Title": job_title,
        "Company_Size": company_size,
        "Location": location,
        "AI_Adoption_Level": ai_adoption_level,
        "Automation_Risk": automation_risk,
        "Remote_Friendly": remote_friendly
    }])

    st.subheader("Input Data")
    st.dataframe(input_data)

if st.button("Predict Job Growth"):
    prediction = model.predict(input_data)[0]

    prediction_proba = model.predict_proba(input_data)[0]
    class_names = model.classes_

    confidence = prediction_proba[class_names.tolist().index(prediction)] * 100

    st.subheader("Prediction Result")

    if prediction == "Growing":
        st.success("This job is predicted to grow.")
    else:
        st.warning("This job is predicted to be stable or slightly at the risk of a decline.")

    st.write(f"Predicted class: **{prediction}**")
    st.write(f"Model confidence: **{confidence:.1f}%**")