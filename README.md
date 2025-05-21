# Hybrid Recommendation System

## Overview

This project implements a hybrid recommendation system that combines **Content-Based Filtering** and **Collaborative Filtering** techniques to recommend movies to users. The system is designed to provide personalized recommendations based on both user preferences (from their ratings) and the content of the movies (such as genres).

## Files in the Repository

* `hybrid.py`: Main script containing the implementation of the hybrid recommendation system.
* `movies.csv`: Dataset containing information about movies, including their genres.
* `ratings.csv`: Dataset containing user ratings for various movies.

## Key Features

1. **Content-Based Filtering**:

   * Utilizes the movie genres to compute similarity between movies using the **TF-IDF Vectorizer** and **Cosine Similarity**.
   * Recommends movies similar to a given movie.

2. **Collaborative Filtering**:

   * Implements **Singular Value Decomposition (SVD)** to predict user ratings for movies.
   * Recommends movies based on user preferences inferred from their past ratings.

3. **Hybrid Model**:

   * Combines Content-Based and Collaborative Filtering techniques using a weighted approach.
   * Allows fine-tuning the balance between the two methods using the `alpha` parameter.

4. **Evaluation Metrics**:

   * Assesses the model using metrics such as RMSE, MAE, Precision, Recall, and F1-Score.

## Key Imports and Their Roles

### Python Libraries

* **Pandas**:
  Used for data manipulation and analysis, such as reading CSV files and handling dataframes.

* **NumPy**:
  Provides numerical computation capabilities, such as matrix operations and statistical calculations.

* **Scikit-learn**:

  * `TfidfVectorizer`: Converts text data (genres) into numerical features for similarity computation.
  * `cosine_similarity`: Computes the similarity between movies based on their genres.
  * `train_test_split`: Splits data into training and testing sets for evaluation.
  * `mean_squared_error`, `mean_absolute_error`: Calculates evaluation metrics for the model.
  * `MinMaxScaler`: Normalizes scores for fair combination in the hybrid model.

* **SciPy**:

  * `svds`: Performs Singular Value Decomposition (SVD) for collaborative filtering.

### Datasets

* **Movies Dataset (`movies.csv`)**:
  Contains movie information including `movieId`, `title`, and `genres`.

* **Ratings Dataset (`ratings.csv`)**:
  Contains user ratings for movies with fields: `userId`, `movieId`, and `rating`.

## Code Explanation

### Functions

1. **Content-Based Filtering**:

   * `content_based_recommend(movie_id, top_n)`: Recommends movies similar to a given movie based on genre similarity.

2. **Collaborative Filtering**:

   * `predict_rating(user_id, movie_id)`: Predicts the rating a user would give to a specific movie.
   * `evaluate_model(test_ratings)`: Evaluates the model's performance using test data.

3. **Hybrid Recommendation**:

   * `hybrid_recommend(user_id, movie_id, alpha, top_n)`: Combines content-based and collaborative filtering to provide recommendations.

4. **Utility Functions**:

   * `get_all_users()`: Retrieves all unique user IDs from the dataset.
   * `get_all_movies()`: Retrieves all movies and their details from the dataset.

## How to Use

1. **Setup**:

   * Ensure `movies.csv` and `ratings.csv` are in the same directory as `hybrid.py`.
   * Install the required Python libraries (e.g., Pandas, NumPy, Scikit-learn, SciPy).

2. **Run the Script**:

   * Load the script in your Python environment.
   * Use the predefined functions to:

     * Get recommendations based on a movie or user.
     * Evaluate the performance of the model.

3. **Example Usage**:

   * Get top 10 movies similar to a given movie:

     ```python
     recommendations = content_based_recommend(movie_id=1, top_n=10)
     print(recommendations)
     ```
   * Predict a user’s rating for a specific movie:

     ```python
     predicted_rating = predict_rating(user_id=1, movie_id=2)
     print(predicted_rating)
     ```
   * Get hybrid recommendations:

     ```python
     hybrid_recs = hybrid_recommend(user_id=1, movie_id=3, alpha=0.7, top_n=5)
     print(hybrid_recs)
     ```

## Goal

The primary goal of this project is to create a robust recommendation system that effectively combines multiple techniques to enhance the accuracy and relevance of recommendations, catering to diverse user needs and preferences.
