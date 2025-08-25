import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Sales Data - Dimensionality Reduction", layout="wide")
st.title("🔻 Dimensionality Reduction using PCA and t-SNE on Sales Data")

# Generate synthetic sales dataset
@st.cache_data
def generate_sales_data(n=300):
    np.random.seed(42)
    data = pd.DataFrame({
        'TV_Ad_Spend': np.random.normal(20000, 4000, n),
        'Radio_Ad_Spend': np.random.normal(10000, 2500, n),
        'Social_Media_Ad_Spend': np.random.normal(15000, 3500, n),
        'Store_Promo_Spend': np.random.normal(12000, 3000, n),
        'Sales_Rep_Spend': np.random.normal(8000, 2000, n),
        'Month': np.random.randint(1, 13, n)
    })
    
    # Assign region as category
    data['Region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n)
    return data

# Load data
df = generate_sales_data()
st.subheader("📊 Sample Sales Dataset")
st.dataframe(df.head())

# Encode region for dimensionality reduction
df_encoded = df.copy()
df_encoded['Region_Code'] = df_encoded['Region'].map({'North':0, 'South':1, 'East':2, 'West':3})

# Features for reduction
features = ['TV_Ad_Spend', 'Radio_Ad_Spend', 'Social_Media_Ad_Spend', 
            'Store_Promo_Spend', 'Sales_Rep_Spend', 'Month', 'Region_Code']

X = df_encoded[features]
X_scaled = StandardScaler().fit_transform(X)

# Visualize original high-dimensional feature space (2D projection)
st.subheader("🔹 Original Data (First 2 Features Only)")
fig1, ax1 = plt.subplots()
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['Region'], palette='tab10', ax=ax1)
ax1.set_xlabel("TV Ad Spend (scaled)")
ax1.set_ylabel("Radio Ad Spend (scaled)")
st.pyplot(fig1)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
st.subheader("🔹 PCA Visualization (2D)")
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca['Region'] = df['Region']
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Region', palette='tab10', ax=ax2)
st.pyplot(fig2)

# t-SNE
st.subheader("🔹 t-SNE Visualization (2D)")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_scaled)
df_tsne = pd.DataFrame(X_tsne, columns=["Dim1", "Dim2"])
df_tsne['Region'] = df['Region']
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='Region', palette='tab10', ax=ax3)
st.pyplot(fig3)

# Summary
st.markdown("---")
st.markdown("✅ **Summary:**")
st.markdown("""
- **PCA** is a linear reduction technique that helps compress the data into components.
- **t-SNE** is nonlinear and great for visualizing clusters or groupings.
- We've used synthetic **Sales Spend data** and visualized how **regions** form patterns after reduction.
""")
