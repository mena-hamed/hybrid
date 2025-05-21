import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split # Used for splitting data for evaluation
from hybrid import content_based_recommend, hybrid_recommend, evaluate_model, movies, ratings


st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title(" Movie Recommendation System")

# --- Sidebar for Search Options ---
st.sidebar.header("🔍 Search Options")

# Select a user ID for recommendations
user_ids = ratings['userId'].unique()
user_id = st.sidebar.selectbox("Select a User ID", user_ids)

# Select a movie title to base recommendations on
movie_titles = movies['title'].tolist()
movie_title = st.sidebar.selectbox("Select a Movie Title", movie_titles)

# Get movieId from the selected movie title
# Added a check to ensure movie_id is found
movie_id_row = movies[movies['title'] == movie_title]
movie_id = movie_id_row['movieId'].values[0] if not movie_id_row.empty else None

# Choose alpha for hybrid weighting (how much to favor collaborative filtering)
alpha = st.sidebar.slider("Hybrid Weight (Higher = More Collaborative)", 0.0, 1.0, 0.5, 0.1)

# Button to get recommendations
if st.sidebar.button("Get Recommendations"):
    if movie_id is not None:
        st.subheader(f" Content-Based Recommendations for '{movie_title}':")
        cb_recs = content_based_recommend(movie_id)
        st.dataframe(cb_recs)

        st.subheader(f" Hybrid Recommendations for User {user_id} based on '{movie_title}':")
        hybrid_recs = hybrid_recommend(user_id, movie_id, alpha)
        st.dataframe(hybrid_recs)
    else:
        st.warning("Please select a valid movie title to get recommendations.")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")

# --- Evaluation Section (New Requirement: Display Evaluation Metrics) ---
st.header("📊 Model Evaluation")
st.write("Assess the performance of the collaborative filtering component of the hybrid model.")
st.info("Note: Evaluation can be computationally intensive for large datasets. The metrics displayed here are based on a random split of your ratings data.")

# Sliders for configuring evaluation parameters
test_size = st.slider("Test Set Size for Evaluation (e.g., 0.2 means 20% of data for testing)", 0.1, 0.5, 0.2, 0.05)
rating_threshold = st.slider("Relevance Threshold for Precision/Recall/F1 (Ratings >= this are 'relevant')", 2.5, 4.5, 3.5, 0.5)

# Button to trigger evaluation
if st.button("Run Evaluation"):
    with st.spinner("Running evaluation... This may take a moment."):
        # Split the entire ratings dataset into training and testing sets for evaluation purposes.
        # The model (SVD) is built on the full dataset, but evaluation uses a separate test set
        # to simulate unseen data.
        _, test_ratings = train_test_split(ratings, test_size=test_size, random_state=42)

        # Call the evaluation function from recommender.py
        metrics = evaluate_model(test_ratings, rating_threshold)

        st.subheader("Evaluation Metrics:")
        # Display metrics using Streamlit's st.metric for a clear overview
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(label="RMSE", value=f"{metrics['RMSE']:.3f}")
        with col2:
            st.metric(label="MAE", value=f"{metrics['MAE']:.3f}")
        with col3:
            st.metric(label="Precision", value=f"{metrics['Precision']:.3f}")
        with col4:
            st.metric(label="Recall", value=f"{metrics['Recall']:.3f}")
        with col5:
            st.metric(label="F1-Score", value=f"{metrics['F1-Score']:.3f}")

        st.success("Evaluation complete!")

 