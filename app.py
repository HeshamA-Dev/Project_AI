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

RAW_DATA_PATH = "data/raw/ai_job_market_insights.csv"
CLEANED_DATA_PATH = "data/cleaned/ai_job_market_insights_cleaned.csv"
MODEL_PATH = "model/job_growth_model.pkl"


@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_cleaned_data():
    return pd.read_csv(CLEANED_DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


raw_df = load_raw_data()
df = load_cleaned_data()
model = load_model()



# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
    "1. Project Kickoff",
    "2. Dataset Exploration",
    "3. Data Cleaning Protocol",
    "4. Visualizations",
    "5. Machine Learning",
    "6. Live Prediction"
    ]
)


# --------------------------------------------------
# Home page
# --------------------------------------------------

if page == "1. Project Kickoff":
    st.title("AI Job Impact Predictor")

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

    st.subheader("My Project Roadmap")

    st.write("""
    1. Load and understand the dataset  
    2. Clean and prepare the data  
    3. Visualize important patterns  
    4. Train a scikit-learn classification model  
    5. Use the model in a Streamlit app for live predictions  
    """)
    
    st.subheader("Technical Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Model Accuracy", "56%")

    st.subheader("Project Links")

    st.markdown("""
    - [GitHub Repository](https://github.com/HeshamA-Dev/Project_AI)
    - [Dataset Source](https://www.kaggle.com/datasets/uom190346a/ai-powered-job-market-insights?resource=download)
    """)

# --------------------------------------------------
# Dataset page
# --------------------------------------------------

elif page == "2. Dataset Exploration":
    st.title("Dataset Overview")

    st.write("""
    This page shows the original dataset and the cleaned version used for visualization and machine learning.
    """)

    st.subheader("Dataset Comparison")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Raw Dataset Rows", raw_df.shape[0])
        st.metric("Raw Dataset Columns", raw_df.shape[1])

    with c2:
        st.metric("Cleaned Dataset Rows", df.shape[0])
        st.metric("Cleaned Dataset Columns", df.shape[1])

    tab1, tab2 = st.tabs([
        "Raw Data Preview",
        "Cleaned Data Preview"
    ])

    with tab1:
        st.subheader("Raw Dataset Preview")
        st.write("""
        This is the original dataset before applying the cleaning pipeline.
        """)
        st.dataframe(raw_df.head(10), use_container_width=True)

    with tab2:
        st.subheader("Cleaned Dataset Preview")
        st.write("""
        This is the cleaned dataset after applying the data cleaning steps.
        """)
        st.dataframe(df.head(10), use_container_width=True)



# --------------------------------------------------
# Data Cleaning page
# --------------------------------------------------

elif page == "3. Data Cleaning Protocol":
    st.title("Data Cleaning Protocol")

    st.write("""
    This page summarizes the main cleaning steps that were applied to the raw dataset.
    Each step shows what was checked and what the final result was.
    """)

    st.subheader("Cleaning Overview")

    with st.expander("1. Missing Values Check", expanded=False):
        missing_values = df.isnull().sum().sum()

        st.write("The dataset was checked for missing values in all columns.")

        if missing_values == 0:
            st.success("Result: No missing values were found.")
        else:
            st.warning(f"Result: {missing_values} missing values were found.")

        st.dataframe(df.isnull().sum().to_frame(name="Missing Values"))

    with st.expander("2. Duplicate Rows Check", expanded=False):
        duplicate_rows = df.duplicated().sum()

        st.write("The dataset was checked for duplicate rows.")

        if duplicate_rows == 0:
            st.success("Result: No duplicate rows were found.")
        else:
            st.warning(f"Result: {duplicate_rows} duplicate rows were found.")

    with st.expander("3. Salary Data Type Conversion", expanded=False):
        st.write("""
        The column `Salary_USD` was converted from float to integer.
        Salary values were rounded before converting them to avoid simply cutting off decimal values.
        """)

        st.success("Result: Salary_USD was rounded and converted to integer values.")


    with st.expander("4. Remote_Friendly Conversion", expanded=False):
        st.write("""
        The column `Remote_Friendly` originally contained `Yes` and `No`.
        It was converted into a binary format for easier analysis and machine learning.
        """)

        st.write("""
        - Yes = 1
        - No = 0
        """)

        st.success("Result: Remote_Friendly was converted from Yes/No into 1/0.")

        remote_table = df["Remote_Friendly"].map({
            1: "Yes",
            0: "No"
        }).value_counts(normalize=True) * 100

        remote_table = remote_table.round(1).to_frame(name="Percentage (%)")

        st.dataframe(remote_table)

    with st.expander("5. Salary Outlier Check", expanded=False):
        st.write("""
        The salary column was checked for possible outliers using the IQR method.
        This method identifies unusually low or high values compared to the main salary distribution.
        """)

        q1 = df["Salary_USD"].quantile(0.25)
        q3 = df["Salary_USD"].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[
            (df["Salary_USD"] < lower_bound) |
            (df["Salary_USD"] > upper_bound)
        ]

        st.write(f"Q1: {q1:.2f}")
        st.write(f"Q3: {q3:.2f}")
        st.write(f"IQR: {iqr:.2f}")
        st.write(f"Lower bound: {lower_bound:.2f}")
        st.write(f"Upper bound: {upper_bound:.2f}")

        st.write(f"Possible salary outliers found: {len(outliers)}")

        st.success("""
        Result: Possible salary outliers were checked but not removed,
        because high or low salaries can still be realistic depending on job title,
        location, company size, industry, or experience level.
        """)

    with st.expander("6. Final Cleaned Dataset", expanded=False):
        st.write("""
        After cleaning, the dataset was saved as a new CSV file and used for visualization,
        machine learning, and the Streamlit application.
        """)

        st.success(f"Result: Final dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

        st.dataframe(df.head())


# --------------------------------------------------
# Visualizations page
# --------------------------------------------------

elif page == "4. Visualizations":
    st.title("Visualizations")

    st.write("""
    This page shows the most important visualizations from the dataset.
    The charts help explain salary distribution, automation risk, job growth, and the relationship between AI adoption and job growth.
    """)

    # --------------------------------------------------
    # 1. Salary Distribution
    # --------------------------------------------------

    st.subheader("Salary Distribution")

    st.write("""
    This histogram shows how salaries are distributed in the dataset.
    It helps identify common salary ranges and possible unusually high or low salary values.
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["Salary_USD"], bins=20, edgecolor="black")
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Salary in USD")
    ax.set_ylabel("Number of Jobs")

    st.pyplot(fig)

    # --------------------------------------------------
    # 2. Automation Risk Distribution
    # --------------------------------------------------

    st.subheader("Automation Risk Distribution")

    st.write("""
    This chart shows how many jobs have low, medium, or high automation risk.
    This is important because automation risk is directly connected to the topic of AI and jobs.
    """)

    risk_counts = df["Automation_Risk"].value_counts()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    risk_counts.plot(kind="bar", ax=ax2, edgecolor="black")
    ax2.set_title("Automation Risk Distribution")
    ax2.set_xlabel("Automation Risk")
    ax2.set_ylabel("Number of Jobs")
    ax2.tick_params(axis="x", rotation=0)

    st.pyplot(fig2)

    # --------------------------------------------------
    # 3. Job Growth Distribution
    # --------------------------------------------------

    st.subheader("Job Growth Distribution")

    st.write("""
    This chart shows the distribution of the original job growth categories.
    This is especially important because job growth is the target variable for the machine learning model.
    """)

    growth_counts = df["Job_Growth_Projection"].value_counts()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    growth_counts.plot(kind="bar", ax=ax3, edgecolor="black")
    ax3.set_title("Job Growth Projection Distribution")
    ax3.set_xlabel("Job Growth Projection")
    ax3.set_ylabel("Number of Jobs")
    ax3.tick_params(axis="x", rotation=0)

    st.pyplot(fig3)

    # --------------------------------------------------
    # 4. Job Growth by AI Adoption Level
    # --------------------------------------------------

    st.subheader("Job Growth Projection by AI Adoption Level")

    st.write("""
    This chart compares job growth categories across different AI adoption levels.
    It helps analyze whether jobs in environments with higher AI adoption show different growth patterns.
    """)

    ai_growth_table = pd.crosstab(
        df["AI_Adoption_Level"],
        df["Job_Growth_Projection"],
        normalize="index"
    ) * 100

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ai_growth_table.plot(kind="bar", ax=ax4, edgecolor="black")

    ax4.set_title("Job Growth Projection by AI Adoption Level")
    ax4.set_xlabel("AI Adoption Level")
    ax4.set_ylabel("Percentage (%)")
    ax4.tick_params(axis="x", rotation=0)
    ax4.legend(title="Job Growth Projection")

    st.pyplot(fig4)
    
# --------------------------------------------------
# Machine Learning page
# --------------------------------------------------
elif page == "5. Machine Learning":
    st.title("Machine Learning Model")

    st.write("""
    The machine learning part of this project is a supervised classification task.
    The goal is to predict whether a job is likely to grow or not.
    """)

    st.subheader("Prediction Target")

    st.write("""
    The original target column was `Job_Growth_Projection`.
    It originally contained three values:
    """)

    st.write("""
    - Growth
    - Stable
    - Decline
    """)

    st.write("""
    For the model, this target was converted into a binary variable:
    """)

    st.write("""
    - Growth = Growing
    - Stable = Not Growing
    - Decline = Not Growing
    """)

    st.subheader("Input Features")

    st.write("""
    The model uses the following features:
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

    st.subheader("Algorithm")

    st.write("""
    I used a Random Forest Classifier from scikit-learn.
    A Random Forest combines many decision trees. In this project, the model uses 100 decision trees.
    Each tree makes a prediction, and the final result is based on majority voting.
    """)

    st.subheader("Preprocessing")

    st.write("""
    Most input features are categorical, so they were converted into numerical values using OneHotEncoder.
    The preprocessing step and the Random Forest model were combined in a scikit-learn Pipeline.
    """)

    st.subheader("Evaluation")

    st.write("""
    The dataset was split using an 80/20 train-test split.
    This means that 80% of the data was used for training and 20% was used for testing.
    """)

    st.metric("Model Accuracy", "56%")

    st.write("""
    The classification report showed that the model predicted `Not Growing` better than `Growing`.
    One reason is that the dataset is imbalanced: there are more `Not Growing` examples than `Growing` examples.
    """)



# --------------------------------------------------
# Live Prediction page
# --------------------------------------------------

elif page == "6. Live Prediction":
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