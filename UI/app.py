import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
with open("../models/final_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset for visualization
@st.cache_data
def load_data():
    return pd.read_csv("../data/heart_disease.csv")

def main():
    st.title("❤️ Heart Disease Prediction App")

    # Sidebar menu
    menu = ["Prediction", "Data Visualization"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prediction":
        st.subheader("Predict Heart Disease")

        # Collect user input
        age = st.number_input("Age", 20, 100, 30)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina",
                3: "Non-anginal Pain",
                4: "Asymptomatic"
            }[x]
        )

        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression (oldpeak)", -2.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST segment", [1, 2, 3])
        ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [3, 6, 7], format_func=lambda x: {
            3: "Normal",
            6: "Fixed Defect",
            7: "Reversible Defect"
        }[x])

        # Prepare features (make sure order matches training)
        features = np.array([[age, sex, cp, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Prediction button
        if st.button("Predict"):
            prediction = model.predict(features)[0]
            if prediction == 1:
                st.error("⚠️ The model predicts **Heart Disease Present**")
            else:
                st.success("✅ The model predicts **No Heart Disease**")

    elif choice == "Data Visualization":
        st.subheader("📊 Explore Heart Disease Dataset")

        df = load_data()
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()

        # Heart Disease distribution
        st.write("### ❤️ Heart Disease Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="num", data=df, ax=ax)
        ax.set_title("Heart Disease Presence (0 --> No, otherwise --> Yes)")
        st.pyplot(fig)

        # Age Distribution
        st.write("### 👤 Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["age"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

        # Chest Pain vs Disease
        st.write("### 💢 Chest Pain Type vs Heart Disease")
        fig, ax = plt.subplots()
        sns.countplot(x="cp", hue="num", data=df, ax=ax)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
