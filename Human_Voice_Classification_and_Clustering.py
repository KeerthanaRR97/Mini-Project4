import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ========== LOAD & CLEAN DATA ==========

@st.cache_data
def load_data():
    df = pd.read_csv("D:/Guvi_Project4/Dataset/vocal_gender_features_new.csv")  # Make sure file path is correct
    df.dropna(inplace=True)  # Clean missing data
    return df

df = load_data()

# ========== DATA PREPROCESSING ==========
X = df.drop('label', axis=1)
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== MODEL TRAINING ==========
clf = RandomForestClassifier(random_state=42)
clf.fit(X_scaled, y)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ========== SIDEBAR NAVIGATION ==========
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏁 Introduction", "📊 EDA", "🤖 Classification & Prediction", "📌 Clustering", "🙋 About Me"])

# ------------------------ INTRODUCTION ------------------------
if page == "🏁 Introduction":
    st.title("🎤 Human Voice Classification and Clustering")
    st.image("D:\Guvi_Project4\Dataset\project4_image.png", caption="Voice Spectrogram Sample",  use_container_width=True)

    st.markdown("""
    ### 📘 Project Summary

    This project uses **machine learning and signal processing** to classify and cluster human voices using features like **pitch**, **MFCCs**, and **spectral characteristics**.

    ### 🧱 Pipeline Overview

    **Data Preparation:**
    - Clean missing data
    - Normalize features using `StandardScaler`

    **EDA:**
    - Visualize pitch, MFCCs, correlations

    **Model Development:**
    - Classification: `Random Forest`, *(extendable to SVM, Neural Nets)*
    - Clustering: `KMeans` with `PCA` visualization

    **Evaluation:**
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - Silhouette Score for clusters

    **Deployment:**
    - `Streamlit` interactive UI

    ### 🛠 Tools Used
    `Python`, `Pandas`, `Scikit-learn`, `Seaborn`, `Matplotlib`, `Streamlit`
    """)

# ------------------------ EDA ------------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    questions = {
        "1️⃣ Gender distribution (Pie Chart)": "q1",
        "2️⃣ Mean pitch variation by gender (Bar Plot)": "q2",
        "3️⃣ MFCC_1_mean distribution by gender (Box Plot)": "q3",
        "4️⃣ Mean Pitch vs RMS Energy (Scatter Plot)": "q4",
        "5️⃣ Feature Correlation (Heatmap)": "q5"
    }

    selected_q = st.selectbox("Choose a question to explore:", list(questions.keys()))

    if questions[selected_q] == "q1":
        st.subheader("Gender Distribution (Pie Chart)")
        labels = ['Female', 'Male']
        sizes = df['label'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    elif questions[selected_q] == "q2":
        st.subheader("Mean Pitch by Gender")
        mean_pitch = df.groupby('label')['mean_pitch'].mean()
        fig, ax = plt.subplots()
        ax.bar(['Female', 'Male'], mean_pitch, color=['pink', 'lightblue'])
        ax.set_ylabel("Mean Pitch")
        st.pyplot(fig)

    elif questions[selected_q] == "q3":
        st.subheader("Box Plot of MFCC_1_mean by Gender")
        df['Gender'] = df['label'].map({0: 'Female', 1: 'Male'})
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Gender', y='mfcc_1_mean', ax=ax)
        st.pyplot(fig)

    elif questions[selected_q] == "q4":
        st.subheader("Scatter Plot: Mean Pitch vs RMS Energy")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df['mean_pitch'], df['rms_energy'], c=df['label'], cmap='coolwarm', alpha=0.7)
        ax.set_xlabel("Mean Pitch")
        ax.set_ylabel("RMS Energy")
        ax.set_title("Pitch vs Energy (by Gender)")
        st.pyplot(fig)

    elif questions[selected_q] == "q5":
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.drop(['label', 'Gender'], axis=1, errors='ignore').corr(), cmap='coolwarm', annot=False, ax=ax)
        st.pyplot(fig)

# ------------------------ CLASSIFICATION ------------------------
elif page == "🤖 Classification & Prediction":
    st.title("🤖 Voice Gender Classification")

    st.markdown("### 🔢 Input Features")
    input_features = ['mean_pitch', 'max_pitch', 'std_pitch', 'rms_energy', 'spectral_kurtosis',
                      'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_1_std', 'mfcc_3_std', 'mean_spectral_flatness']
    input_data = []

    for feat in input_features:
        val = st.number_input(f"{feat}", format="%.6f")
        input_data.append(val)

    if st.button("Predict Gender"):
        input_df = pd.DataFrame([input_data], columns=input_features).reindex(columns=X.columns, fill_value=0)
        input_scaled = scaler.transform(input_df)
        pred = clf.predict(input_scaled)[0]
        prob = clf.predict_proba(input_scaled)[0]
        st.success(f"Predicted: {'Male' if pred == 1 else 'Female'} ({np.max(prob):.2%} confidence)")

    # Model Evaluation
    st.subheader("📋 Evaluation Metrics")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    clf_eval = RandomForestClassifier()
    clf_eval.fit(X_train, y_train)
    y_pred = clf_eval.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

# ------------------------ CLUSTERING ------------------------
elif page == "📌 Clustering":
    st.title("📌 Clustering of Voices (KMeans)")

    st.subheader("🔍 Elbow Method: Best K for KMeans")
    silhouette_scores = []
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(range(2, 10), silhouette_scores, marker='o')
    ax.set_title("Elbow Curve (Silhouette Score)")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Silhouette Score")
    st.pyplot(fig)

    k_val = st.slider("Select number of clusters", 2, 10, 3)
    model = KMeans(n_clusters=k_val, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)
    st.write(f"🧮 Silhouette Score for K={k_val}: {silhouette_score(X_scaled, cluster_labels):.3f}")

    st.subheader("🖼 PCA Cluster Visualization")
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.8)
    ax2.set_title("PCA Projection of Clusters")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    st.pyplot(fig2)

# ------------------------ ABOUT ME ------------------------
elif page == "🙋 About Me":
    st.title("🙋 R R KEERTANA")
    st.markdown("""
    ### 👩‍💻 Profile
    - **Degree**: B.E. in Computer Science and Engineering  
    - **Interests**: Machine Learning, Audio/Speech Processing, Data Visualization

    ### 🔗 Connect
    - [LinkedIn](https://www.linkedin.com)
    - [GitHub](https://github.com)

    _Thank you for exploring my project!_
    """)

