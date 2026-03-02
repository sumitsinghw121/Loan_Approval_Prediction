import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ----------------------
# Load Model
# ----------------------
with open("rf_model (2).pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")

st.title("💰 Loan Approval Prediction System")
st.markdown("AI-powered system to predict loan approval status")

# ----------------------
# Load Dataset (For Accuracy + Graphs)
# ----------------------
df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip().str.lower()

# Clean categorical columns
df["education"] = df["education"].astype(str).str.strip()
df["self_employed"] = df["self_employed"].astype(str).str.strip()
df["loan_status"] = df["loan_status"].astype(str).str.strip()

# Encode categorical values
df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

# Drop loan_id if exists
if "loan_id" in df.columns:
    df.drop("loan_id", axis=1, inplace=True)

# Remove missing rows
df = df.dropna()

# ----------------------
# Calculate Accuracy
# ----------------------
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

# ----------------------
# Top Statistics Section
# ----------------------
col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", f"{round(accuracy*100,2)}%")
col2.metric("Features Used", X.shape[1])
col3.metric("Algorithm", "Random Forest")

st.divider()

# ----------------------
# Prediction Section
# ----------------------
st.subheader("🔍 Predict Loan Status")

col1, col2, col3 = st.columns(3)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    income_annum = st.number_input("Annual Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term", min_value=0)

with col3:
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Encode input
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

input_data = np.array([[ 
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]])

if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

st.divider()

# ----------------------
# Dataset Insights Section
# ----------------------
st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

# ----------------------
# FIG 1 - Loan Status Distribution
# ----------------------
with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(x=df["loan_status"], ax=ax1)
    ax1.set_xticklabels(["Rejected", "Approved"])
    ax1.set_title("Loan Status Count")
    st.pyplot(fig1)

# ----------------------
# FIG 2 - Correlation Heatmap
# ----------------------
with col2:
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    ax2.set_title("Feature Correlation")
    st.pyplot(fig2)

st.divider()

# ----------------------
# FIG 3 - Feature Distribution
# ----------------------
st.subheader("📈 Feature Distributions by Loan Status")

feature = st.selectbox(
    "Select Feature",
    numeric_df.columns.drop("loan_status")
)

selected_status = st.multiselect(
    "Select Loan Status to Compare",
    df["loan_status"].unique(),
    default=df["loan_status"].unique()
)

filtered_df = df[df["loan_status"].isin(selected_status)]

fig3, ax3 = plt.subplots(figsize=(4,3))

for status in selected_status:
    sns.histplot(
        filtered_df[filtered_df["loan_status"] == status][feature],
        label=str(status),
        kde=True,
        bins=20,
        ax=ax3
    )

ax3.set_title(f"{feature} Distribution")
ax3.legend()
st.pyplot(fig3)

st.divider()

# ----------------------
# FIG 4 & FIG 5 - Dual Feature Comparison
# ----------------------
st.subheader("📊 Compare Two Features by Loan Status")

numeric_columns = numeric_df.columns.drop("loan_status")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.selectbox("Select First Feature", numeric_columns, key="f1")

with col2:
    feature2 = st.selectbox("Select Second Feature", numeric_columns, key="f2")

selected_status2 = st.multiselect(
    "Select Loan Status",
    df["loan_status"].unique(),
    default=df["loan_status"].unique(),
    key="status2"
)

filtered_df2 = df[df["loan_status"].isin(selected_status2)]

col1, col2 = st.columns(2)

with col1:
    fig4, ax4 = plt.subplots(figsize=(5,3))
    for status in selected_status2:
        sns.histplot(
            filtered_df2[filtered_df2["loan_status"] == status][feature1],
            kde=True,
            bins=20,
            label=str(status),
            ax=ax4
        )
    ax4.set_title(f"{feature1} Distribution")
    ax4.legend()
    st.pyplot(fig4)

with col2:
    fig5, ax5 = plt.subplots(figsize=(5,3))
    for status in selected_status2:
        sns.histplot(
            filtered_df2[filtered_df2["loan_status"] == status][feature2],
            kde=True,
            bins=20,
            label=str(status),
            ax=ax5
        )
    ax5.set_title(f"{feature2} Distribution")
    ax5.legend()
    st.pyplot(fig5)