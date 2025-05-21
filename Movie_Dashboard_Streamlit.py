import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# Load data
full_df = pd.read_csv("full_merged_cleaned.csv")

# Preprocessing
ratings_per_user = full_df.groupby("userId")["rating"].count()
top_tags = full_df['tag'].value_counts().head(20)
full_df['genre_list'] = full_df['genres'].str.split('|')
genre_exploded = full_df.explode('genre_list')
top_genres = genre_exploded['genre_list'].value_counts().head(15)

# User-item matrix
sample_df = full_df[['userId', 'movieId', 'rating']]
user_item_matrix = sample_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Clustering
svd = TruncatedSVD(n_components=20, random_state=42)
user_features = svd.fit_transform(user_item_matrix)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(user_features)
user_clusters = pd.DataFrame({'userId': user_item_matrix.index, 'cluster': clusters})

# Streamlit App
st.set_page_config(page_title="🎬 Interactive Movie Insights", layout="wide")
st.title("🎬 Interactive Movie Insights Dashboard")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Movie Ratings by Cluster & User",
    "Ratings per User",
    "Top Tags Used",
    "Top Genres Overall",
    "User Cluster Distribution",
    "Top Genres by Cluster"
])

# Tab 1: Movie Ratings by Cluster & User
with tab1:
    st.subheader("🎯 Movie Ratings by Cluster and User")
    st.markdown("""
    This section presents individual user rating behaviour within each cluster. 
    Selecting a cluster and then a user displays the specific movies rated by that user.
    """)

    col1, col2 = st.columns([1, 3])

    with col1:
        cluster_val = st.selectbox("Choose a Cluster", options=sorted(user_clusters['cluster'].unique()))
        users_in_cluster = user_clusters[user_clusters['cluster'] == cluster_val]['userId']
        user_val = st.selectbox("Choose a User", options=sorted(users_in_cluster))

    with col2:
        if user_val:
            user_data = sample_df[sample_df['userId'] == user_val].merge(
                full_df[['movieId', 'title']].drop_duplicates(), on='movieId')
            fig = px.bar(user_data, x='title', y='rating',
                         title=f"⭐ Movie Ratings by User {user_val}",
                         labels={'title': 'Movie', 'rating': 'Rating'},
                         color_discrete_sequence=['lightblue'])
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Ratings per User
with tab2:
    st.subheader("📊 Ratings per User")
    fig = px.histogram(ratings_per_user, nbins=30, title='Number of Ratings per User',
                       labels={'value': 'Ratings', 'count': 'Users'},
                       color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("This histogram shows the distribution of how many movies each user has rated.")

# Tab 3: Top Tags
with tab3:
    st.subheader("🏷️ Top Tags Used")
    fig = px.bar(x=top_tags.values, y=top_tags.index, orientation='h',
                 title='Top 20 Tags Used', labels={'x': 'Frequency', 'y': 'Tags'},
                 color=top_tags.values, color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Tags provide insight into user preferences and can be used for content-based filtering.")

# Tab 4: Top Genres
with tab4:
    st.subheader("🎵 Top Genres Overall")
    fig = px.bar(x=top_genres.values, y=top_genres.index, orientation='h',
                 title='Top Genres', labels={'x': 'Count', 'y': 'Genre'},
                 color=top_genres.values, color_continuous_scale='Viridis',
                 text=[f'{(v / top_genres.sum() * 100):.2f}%' for v in top_genres.values])
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("This chart shows the most frequent genres and their relative popularity.")

# Tab 5: User Cluster Distribution
with tab5:
    st.subheader("👥 User Cluster Distribution")
    fig = px.histogram(user_clusters.astype({'cluster': str}), x='cluster',
                       title='User Distribution by Cluster', labels={'cluster': 'Cluster ID'},
                       color='cluster', color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Clusters group users with similar movie rating behaviors.")

# Tab 6: Top Genres by Cluster
with tab6:
    st.subheader("🎯 Top Genres by Cluster")
    selected_cluster = st.selectbox("Select Cluster:", sorted(user_clusters['cluster'].unique()))

    clustered_ratings = sample_df.merge(user_clusters, on='userId')
    metadata = full_df[['movieId', 'genres']].drop_duplicates()
    clustered_genres = clustered_ratings.merge(metadata, on='movieId')
    clustered_genres['genre_list'] = clustered_genres['genres'].str.split('|')
    exploded = clustered_genres.explode('genre_list')
    filtered = exploded[exploded['cluster'] == selected_cluster]
    genre_counts = filtered['genre_list'].value_counts().head(10)

    fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                 title=f'Top Genres in Cluster {selected_cluster}', labels={'x': 'Count', 'y': 'Genre'},
                 color=genre_counts.values, color_continuous_scale='Sunset',
                 text=[f'{(v / genre_counts.sum() * 100):.2f}%' for v in genre_counts.values])
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"Users in Cluster {selected_cluster} predominantly watch these genres. This helps guide personalised content suggestions.")
