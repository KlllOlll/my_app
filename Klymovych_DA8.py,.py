# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Функції для завантаження та обробки даних
@st.cache_data
def load_data():
    df = pd.read_csv('Cust_Spend.csv')
    return df.dropna()

def preprocess_data(df):
    df = df.copy()
    df = df[df['Customer Age'] < 100]  # прибираємо аномалії за віком
    df = df[(df['Revenue'] < 10000) & (df['Cost'] < 10000)]  # прибираємо викиди
    df['Unit Price'] = df['Revenue'] / df['Quantity']
    df['Unit Cost'] = df['Cost'] / df['Quantity']
    return df

# Інтерфейс користувача
def main():
    st.set_page_config(
        page_title="Customer Analytics Dashboard",
        layout="wide",
        menu_items={'About': "Аналіз поведінки клієнтів"}
    )

    # Завантаження даних
    df = load_data()
    processed_df = preprocess_data(df)

    # Бічна панель фільтрів
    with st.sidebar:
        st.title("🔍 Фільтри")
        selected_years = st.multiselect(
            "Роки",
            options=df['Year'].unique(),
            default=df['Year'].unique()
        )
        selected_gender = st.selectbox("Стать", df['Customer Gender'].unique())
        show_clusters = st.checkbox("Показати кластеризацію")

    # Фільтрація даних
    filtered_df = processed_df[
        (processed_df['Year'].isin(selected_years)) &
        (processed_df['Customer Gender'] == selected_gender)
    ]

    st.title("📈 Аналіз покупок клієнтів")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Загальний дохід", f"${filtered_df['Revenue'].sum():,.0f}")
    with col2:
        st.metric("Середній вік", f"{filtered_df['Customer Age'].mean():.1f} років")
    with col3:
        st.metric("Кількість транзакцій", filtered_df.shape[0])

    tab1, tab2, tab3 = st.tabs(["Розподіл віку", "Топ категорії", "Кластеризація"])

    with tab1:
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['Customer Age'], bins=20, kde=True)
        st.pyplot(fig)

    with tab2:
        category_counts = filtered_df['Product Category'].value_counts()
        fig = px.bar(category_counts, title="Розподіл по категоріям")
        st.plotly_chart(fig, use_container_width=True)

    if show_clusters:
        with tab3:
            st.header("Аналіз кластерів")
            algorithm = st.selectbox("Алгоритм кластеризації", ["K-Means", "Ієрархічна", "DBSCAN"])

            numeric_cols = filtered_df.select_dtypes(include='number').columns
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(filtered_df[numeric_cols])

            if algorithm == "K-Means":
                n_clusters = st.slider("Кількість кластерів", 2, 10, 4)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(scaled_data)

                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(scaled_data)

                fig = px.scatter(
                    x=pca_data[:, 0], y=pca_data[:, 1], 
                    color=clusters.astype(str), title="K-Means Clustering (PCA)"
                )
                st.plotly_chart(fig, use_container_width=True)

                df_clustered = filtered_df.copy()
                df_clustered['Cluster'] = clusters
                cluster_summary = df_clustered.groupby('Cluster').mean(numeric_only=True)

                st.subheader("Профілі кластерів")
                st.dataframe(cluster_summary)

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(cluster_summary.T, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0.5)
                st.pyplot(fig)

                st.write(f"Silhouette Score: {silhouette_score(scaled_data, clusters):.2f}")
                st.write(f"Davies-Bouldin Index: {davies_bouldin_score(scaled_data, clusters):.2f}")

            elif algorithm == "Ієрархічна":
                st.warning("Ієрархічна кластеризація ще не реалізована")
            elif algorithm == "DBSCAN":
                st.warning("DBSCAN ще не реалізований")

if __name__ == "__main__":
    main()