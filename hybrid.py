import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler # For normalization in hybrid scores

# Load Data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# --- Data Preprocessing (Requirement 1: Handle missing values and ensure data consistency) ---
# Fill NaN genres with an empty string to prevent errors in TF-IDF vectorization
movies['genres'] = movies['genres'].fillna('')

# Ensure movieId and userId columns are of integer type for consistent merging/indexing
movies['movieId'] = movies['movieId'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)
ratings['userId'] = ratings['userId'].astype(int)

# Preprocess genres for TF-IDF: replace '|' with space for better tokenization
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Content-Based Filtering Setup
# Initialize TF-IDF Vectorizer to convert text genres into numerical features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres']) # Fit and transform genres
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) # Compute cosine similarity between all movies
movie_id_to_idx = pd.Series(movies.index, index=movies['movieId']) # Map movie IDs to their DataFrame index

def content_based_recommend(movie_id, top_n=10):
    """
    Generates content-based movie recommendations.
    Args:
        movie_id (int): The ID of the movie to base recommendations on.
        top_n (int): The number of top recommendations to return.
    Returns:
        pd.DataFrame: A DataFrame of recommended movies.
    """
    if movie_id not in movie_id_to_idx:
        return pd.DataFrame() # Return empty DataFrame if movie ID is not found
    idx = movie_id_to_idx[movie_id] # Get the internal index of the movie
    sim_scores = list(enumerate(cosine_sim[idx])) # Get similarity scores for this movie with all others
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Sort by similarity score
    sim_scores = sim_scores[1:top_n+1] # Exclude the movie itself (first element) and take top_n
    movie_indices = [i[0] for i in sim_scores] # Get the internal indices of recommended movies
    return movies.iloc[movie_indices][['movieId', 'title', 'genres']] # Return movie details

# Collaborative Filtering Setup
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)} # Map user IDs to internal indices
movie_id_to_idx_mat = {mid: i for i, mid in enumerate(movie_ids)} # Map movie IDs to internal matrix indices

num_users = len(user_ids)
num_movies = len(movie_ids)
rating_matrix = np.zeros((num_users, num_movies)) # Initialize user-item rating matrix

# Populate the rating matrix
for row in ratings.itertuples():
    uidx = user_id_to_idx[row.userId]
    midx = movie_id_to_idx_mat[row.movieId]
    rating_matrix[uidx, midx] = row.rating

# Normalize the rating matrix by subtracting user's mean rating
user_ratings_mean = np.mean(rating_matrix, axis=1).reshape(-1, 1)
rating_matrix_norm = rating_matrix - user_ratings_mean

# Perform Singular Value Decomposition (SVD)
# k=50 specifies the number of singular values/vectors to compute
U, sigma, Vt = svds(rating_matrix_norm, k=50)
sigma = np.diag(sigma) # Convert sigma (singular values) to a diagonal matrix

# Reconstruct the full predicted rating matrix
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean

def predict_rating(user_id, movie_id):
    """
    Predicts the rating a user would give to a specific movie using collaborative filtering.
    Args:
        user_id (int): The ID of the user.
        movie_id (int): The ID of the movie.
    Returns:
        float: The predicted rating.
    """
    if user_id not in user_id_to_idx or movie_id not in movie_id_to_idx_mat:
        # If user or movie not seen in training data, return the global average rating
        return np.mean(ratings['rating'])
    uidx = user_id_to_idx[user_id]
    midx = movie_id_to_idx_mat[movie_id]
    return all_user_predicted_ratings[uidx, midx]

def hybrid_recommend(user_id, movie_id, alpha=0.5, top_n=10):
    """
    Generates hybrid movie recommendations combining content-based and collaborative filtering.
    Args:
        user_id (int): The ID of the user.
        movie_id (int): The ID of the movie to base recommendations on.
        alpha (float): Weight for collaborative filtering (0.0 to 1.0).
                       Higher alpha means more weight on collaborative filtering.
        top_n (int): The number of top recommendations to return.
    Returns:
        pd.DataFrame: A DataFrame of recommended movies.
    """
    if movie_id not in movie_id_to_idx:
        return pd.DataFrame() # Return empty DataFrame if movie ID is not found

    idx = movie_id_to_idx[movie_id]
    content_scores = cosine_sim[idx] # Content-based similarity scores

    # Get collaborative scores for all movies for the given user
    collab_scores = [predict_rating(user_id, mid) for mid in movies['movieId']]
    collab_scores = np.array(collab_scores)

    # Normalize scores to a 0-1 range for fair weighting
    scaler = MinMaxScaler()
    # Reshape for scaler, then flatten back
    content_scores_norm = scaler.fit_transform(content_scores.reshape(-1, 1)).flatten()
    collab_scores_norm = scaler.fit_transform(collab_scores.reshape(-1, 1)).flatten()

    # Combine scores using weighted averaging
    hybrid_scores = alpha * collab_scores_norm + (1 - alpha) * content_scores_norm
    indices = hybrid_scores.argsort()[::-1] # Get indices that would sort the scores in descending order
    
    # Filter out the input movie itself and take the top_n recommendations
    top_indices = [i for i in indices if movies.iloc[i]['movieId'] != movie_id][:top_n]

    return movies.iloc[top_indices][['movieId', 'title', 'genres']]

def get_all_users():
    """Returns a sorted list of unique user IDs."""
    return sorted(ratings['userId'].unique())

def get_all_movies():
    """Returns a DataFrame of all movie IDs and titles, sorted by title."""
    return movies[['movieId', 'title']].sort_values('title')

# --- Evaluation Metrics (Requirement 6: Assess performance using RMSE, MAE, Precision, Recall, F1-Score) ---

def evaluate_model(test_ratings, rating_threshold=3.5):
    """
    Evaluates the collaborative filtering component of the model using RMSE, MAE,
    Precision, Recall, and F1-Score.
    Note: This evaluation focuses on the collaborative filtering part (predict_rating)
    as it's the component that produces predicted ratings for direct comparison.
    Args:
        test_ratings (pd.DataFrame): DataFrame containing user, movie, and actual ratings for testing.
        rating_threshold (float): Threshold to convert continuous ratings into binary relevance for P/R/F1.
                                  A rating >= threshold is considered 'relevant'.
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    actual_ratings = []
    predicted_ratings = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through test ratings to get predictions from the collaborative model
    for index, row in test_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']

        # Get prediction using the collaborative filtering part
        predicted_rating = predict_rating(user_id, movie_id)

        actual_ratings.append(actual_rating)
        predicted_ratings.append(predicted_rating)

        # For Precision, Recall, F1, we convert ratings to binary relevance
        is_actual_relevant = actual_rating >= rating_threshold
        is_predicted_relevant = predicted_rating >= rating_threshold

        if is_actual_relevant and is_predicted_relevant:
            true_positives += 1
        elif not is_actual_relevant and is_predicted_relevant:
            false_positives += 1
        elif is_actual_relevant and not is_predicted_relevant:
            false_negatives += 1

    # Calculate RMSE and MAE for continuous rating predictions
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    mae = mean_absolute_error(actual_ratings, predicted_ratings)

    # Calculate Precision, Recall, and F1-Score for binary relevance
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    }