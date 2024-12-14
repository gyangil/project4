import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import streamlit as st

# Load the rating matrix from a CSV file
# This matrix contains users as rows and movies as columns
rating_matrix = pd.read_csv("rating_matrix.csv", index_col=0)  # Assuming rows are users, columns are movies

# Step 1: Normalize the rating matrix
def normalize_matrix(matrix):
    """
    Normalize the matrix by centering each row (user).
    Ensures that user biases (e.g., some users always rate higher or lower) are accounted for.
    """
    return matrix.sub(matrix.mean(axis=1, skipna=True), axis=0)

# Step 2: Compute cosine similarity
def compute_similarity(matrix):
    """
    Compute the cosine similarity for movies with transformation.
    Calculates how similar each pair of movies is based on user ratings.
    The similarity values are transformed to the range [0, 1] for convenience.
    """
    movie_matrix = matrix.T  # Transpose to make rows as movies
    similarity = pd.DataFrame(
        cosine_similarity(movie_matrix.fillna(0)),
        index=movie_matrix.index,
        columns=movie_matrix.index
    )
    return (1 + similarity) / 2  # Transform similarity to [0, 1]

# Step 3: Filter low-count similarities
def filter_low_counts(similarity_matrix, rating_matrix, min_shared=3):
    """
    Mask similarities with fewer than min_shared ratings.
    Ensures that the similarity calculation is based on enough data to be meaningful.
    """
    movie_matrix = rating_matrix.T.notna().astype(int)
    shared_counts = movie_matrix.T @ movie_matrix
    filtered = similarity_matrix.copy()
    filtered[shared_counts < min_shared] = np.nan
    return filtered

# Step 4: Keep top 30 similarities
def top_k_similarity(sim_matrix, k=30):
    """
    Keep only the top-k similarities for each movie.
    Reduces computational complexity and focuses on the most relevant similarities.
    """
    return sim_matrix.apply(
        lambda row: row[row.nlargest(k).index].reindex(sim_matrix.columns, fill_value=np.nan), axis=1
    )

# Step 5: Define the myIBCF function
def myIBCF(new_user_ratings, similarity_matrix, max_movies=100):
    """
    Generate movie recommendations for a new user.
    Limit the similarity matrix to only `max_movies` columns to save memory.
    Predictions are made based on the similarity scores and the ratings provided by the user.
    """
    limited_similarity = similarity_matrix.iloc[:, :max_movies]
    """
    Generate movie recommendations for a new user.
    Predictions are made based on the similarity scores and the ratings provided by the user.
    """
    predictions = {}
    for movie in similarity_matrix.index:
        sim_scores = similarity_matrix[movie].dropna()
        rated_movies = new_user_ratings[new_user_ratings.notna()]
        overlapping_movies = sim_scores.index.intersection(rated_movies.index)
        if overlapping_movies.empty:
            predictions[movie] = np.nan
        else:
            weights = sim_scores[overlapping_movies]
            ratings = rated_movies[overlapping_movies]
            predictions[movie] = (weights @ ratings) / weights.sum()
    return pd.Series(predictions)

# Preprocess data
start_time = datetime.now()
normalized_ratings = normalize_matrix(rating_matrix)
raw_similarity = compute_similarity(normalized_ratings)
filtered_similarity = filter_low_counts(raw_similarity, rating_matrix)
top_30_similarity = top_k_similarity(filtered_similarity)
st.write(f"Data preprocessing completed in: {(datetime.now() - start_time).seconds / 60} minutes")

# Streamlit App
st.title("Movie Recommendation System")
st.header("Rate Movies and Get Recommendations")

# User inputs for rating movies
movie_sample = rating_matrix.columns[:100]  # Limit to 100 movies for optimization  # Select a sample of movies for simplicity
user_ratings = {}
st.write("Please rate the following movies:")

for movie in movie_sample:
    user_ratings[movie] = st.slider(f"Rate {movie}", min_value=0, max_value=5, value=0)

# Convert user ratings to a pandas Series
new_user = pd.Series(user_ratings)
new_user = new_user.replace(0, np.nan)  # Replace 0 ratings with NaN

# Generate recommendations
if st.button("Get Recommendations"):
    start_time = datetime.now()
    recommendations = myIBCF(new_user, top_30_similarity, max_movies=100)
    st.write(f"Recommendations generated in: {(datetime.now() - start_time).seconds / 60} minutes")
    st.write("Top 10 Recommendations:")
    st.table(recommendations.nlargest(10))
