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

# Функція завантаження
@st.cache_data
def load_data():
    df = pd.read_csv('Cust_Spend.csv')
    return df.dropna()

# Функція обробки
def preprocess_data(df):
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df['Year'] = df['Date'].dt.year
        df['Invoice Date'] = df['Date']  # для сумісності
    else:
        st.error("Стовпець 'Date' не знайдено в даних!")
        return pd.DataFrame()

    df = df[df['Customer Age'] > 0]
    df = df[df['Revenue'] >= 0]
    return df


# Інтерфейс
def main():
    st.set_page_config(
        page_title="Customer Analytics Dashboard",
        layout="wide",
        menu_items={'About': "Аналіз поведінки клієнтів"}
    )

    df = load_data()
    processed_df = preprocess_data(df)

    with st.sidebar:
        st.title("🔍 Фільтри")
        selected_years = st.multiselect(
            "Роки",
            options=sorted(df['Year'].unique()),
            default=sorted(df['Year'].unique())
        )
        selected_gender = st.selectbox("Стать", df['Customer Gender'].unique())
        show_clusters = st.checkbox("Показати кластеризацію")

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
        sns.histplot(filtered_df['Customer Age'], bins=20, kde=True, ax=ax)
        ax.set_title("Розподіл віку клієнтів")
        st.pyplot(fig)

    with tab2:
        category_counts = filtered_df['Product Category'].value_counts()
        fig = px.bar(category_counts, title="Розподіл по категоріям", labels={'value': 'Кількість', 'index': 'Категорія'})
        st.plotly_chart(fig, use_container_width=True)

    if show_clusters:
        with tab3:
            st.header("Аналіз кластерів")

            algorithm = st.selectbox(
                "Алгоритм кластеризації",
                ["K-Means", "Ієрархічна", "DBSCAN"]
            )

            numeric_cols = filtered_df.select_dtypes(include='number').columns
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(filtered_df[numeric_cols])

            if algorithm == "K-Means":
                n_clusters = st.slider("Кількість кластерів", 2, 10, 4)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(scaled_data)

            elif algorithm == "Ієрархічна":
                n_clusters = st.slider("Кількість кластерів", 2, 10, 4)
                model = AgglomerativeClustering(n_clusters=n_clusters)
                clusters = model.fit_predict(scaled_data)

            elif algorithm == "DBSCAN":
                eps = st.slider("eps (радіус)", 0.1, 5.0, 1.0)
                min_samples = st.slider("Мін. кількість точок у кластері", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(scaled_data)

            # PCA
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            fig = px.scatter(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                color=clusters.astype(str),
                title=f"Кластеризація методом {algorithm}",
                labels={"x": "PCA1", "y": "PCA2", "color": "Кластер"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Метрики
            if len(set(clusters)) > 1 and -1 not in set(clusters):  # DBSCAN може створити 1 кластер або -1 (шум)
                st.success(f"Silhouette Score: {silhouette_score(scaled_data, clusters):.2f}")
                st.info(f"Davies-Bouldin Index: {davies_bouldin_score(scaled_data, clusters):.2f}")
            else:
                st.warning("Недостатньо кластерів для обчислення метрик (або всі точки віднесено до шуму).")

            # Профілювання
            st.subheader("Профілювання кластерів")
            filtered_df['Cluster'] = clusters
            cluster_profile = filtered_df.groupby('Cluster')[numeric_cols].mean()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cluster_profile, cmap="YlGnBu", annot=True, fmt=".1f", ax=ax)
            ax.set_title("Середні значення змінних за кластерами")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
