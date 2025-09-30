# Comprehensive Machine Learning Full Pipeline on Heart Disease (UCI Dataset)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-brightgreen)  

---

## 📌 Overview
This project applies **machine learning** techniques using **scikit-learn** to predict and analyze **heart disease**.  
The end-to-end pipeline includes:  
- Data preprocessing & cleaning  
- Feature selection & dimensionality reduction (PCA)  
- Supervised learning (classification models)  
- Unsupervised learning (clustering)  
- Hyperparameter tuning for optimization  
- Model export and deployment with **Streamlit UI** (optional: Ngrok)  
- Hosting on GitHub for reproducibility and collaboration  

---

## 🎯 Objectives
- Perform **data preprocessing & cleaning** (handle missing values, encoding, scaling).  
- Apply **dimensionality reduction (PCA)** to retain essential variance.  
- Implement **feature selection** (RFE, Chi-Square, Random Forest importance).  
- Train **classification models**: Logistic Regression, Decision Trees, Random Forest, SVM.  
- Apply **clustering methods**: K-Means, Hierarchical Clustering.  
- Optimize models using **GridSearchCV** and **RandomizedSearchCV**.  
- Deploy a **Streamlit UI** for real-time predictions & visualization.  
- Share via **Ngrok** for live demo access.  

---

## 🛠️ Tools & Libraries
- Python 3.x  
- scikit-learn, Pandas, NumPy  
- Matplotlib, Seaborn  
- TensorFlow/Keras *(optional)*  
- Streamlit (for deployment)  
- Ngrok (for sharing app)  

---

## 🔄 Workflow & Deliverables

### 1️⃣ Data Preprocessing & Cleaning  
✔️ Deliverable: Cleaned dataset  

### 2️⃣ Dimensionality Reduction (PCA)  
✔️ Deliverable: PCA-transformed dataset  

### 3️⃣ Feature Selection  
✔️ Deliverable: Reduced dataset with selected features  

### 4️⃣ Supervised Learning – Classification  
✔️ Deliverable: Trained models with metrics  

### 5️⃣ Unsupervised Learning – Clustering  
✔️ Deliverable: Clustering models & results  

### 6️⃣ Hyperparameter Tuning  
✔️ Deliverable: Optimized best-performing model  

### 7️⃣ Model Export & Deployment  
✔️ Deliverable: Deployed interactive ML app  

---

## 📊 Results

### 🔹 Supervised Learning  
- Logistic Regression Accuracy: **0.90**  
- Random Forest Accuracy: **0.883**  
- Decision Tree Accuracy: **0.733**  
- SVM Accuracy: **0.883**  
- **Best Tuned Model Accuracy:** **0.92**  

Confusion Matrix (Tuned Model):  
```
[[33  3]
 [ 2 22]]
```

### 🔹 Unsupervised Learning  
- K-Means Silhouette Score: **0.48**  
- Cluster 0: lower heart disease probability (~28%)  
- Cluster 1: higher heart disease probability (~71%)  

---

🚀 Streamlit Application
The project includes an interactive Streamlit app for real-time predictions.

**To run the app:**
```bash
cd UI
streamlit run app.py
---

## 📂 Project Structure
```
Heart_Disease_Project/
│── data/
│   └── heart_disease.csv
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│── models/
│   └── final_model.pkl
│── ui/
│   └── app.py
│── deployment/
│   └── ngrok_setup.txt
│── results/
│   └── evaluation_metrics.txt
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 🚀 Running the Project

### Install dependencies  
```bash
pip install -r requirements.txt
```

### Run Jupyter notebooks  
```bash
jupyter notebook
```

### Launch the Streamlit App  
```bash
streamlit run ui/app.py
```

### Optional: Share via Ngrok  
```bash
ngrok http 8501
```

---

## 📌 Dataset  
- **UCI Heart Disease Dataset**  
  [Link to Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)  

