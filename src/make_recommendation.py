""" This scripts allows you to us the trained ALS model to recommend movies to a user.

Example using (in terminal) : 
----------------------------
$ python3 scr/make_recommendation.py --user_id 471 --num_recommendations 5

"""

# Importing libraries

import argparse
import os
import logging
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Logging configuration
logging.basicConfig(filename="logs/recommendations.log", level=logging.INFO,
                    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")

# Modules for ALS model
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# Configuration of a Spark session 
spark = SparkSession.builder.appName("ALS").getOrCreate()

# Loading the movies database
from data_cleaning import cleaning

data = cleaning()

logging.info("Data correctly loaded.")

def recommend_movies(user_id : int, num_recommendations : int, movie_data : pyspark.sql.dataframe.DataFrame = data, model_file = "src/als_model"):
    """
    This function uses an ALS model to recommend movies for a given user.

    Args:
    --------
    - user_id : The user ID for whom recommendations should be generated
    - num_recommendations : The number of recommendations desired
    - movie_data : DataFrame containing movie data
    - model_file : The path to the pickle file containing the trained ALS model.

    Returns:
    --------
    - A list of tuples containing titles and genres of recommended movies for the specified user
    """

    from pyspark.ml.recommendation import ALSModel

    if not os.path.exists(model_file):
        logging.error(f"Model file '{model_file}' does not exist.")
        return []

    try:
        # Load the model from the path
        model = ALSModel.load("src/als_model")
        logging.info("Model successfully loaded")
    except Exception as e:
        logging.error(f"Error loading model from '{model_file}': {e}")
        return []

    # Generate recommendations for the specified user
    user_recommendations = model.recommendForUserSubset(
        movie_data.select("userId").where(col("userId") == user_id), num_recommendations)
    
    # Retrieve titles and genres of recommended movies
    recommendations = user_recommendations.select("recommendations").collect()[0][0]
    
    # Filter movie data to retrieve titles and genres of recommended movies
    recommended_movies = []
    for movie in recommendations:
        movie_id = movie['movieId']
        movie_info = movie_data.select("title", "genres").where(col("movieId") == movie_id).collect()[0]
        recommended_movies.append((movie_info['title'], movie_info['genres']))

    # Display recommended movies
    logging.info("*****|*****|*****")
    logging.info(f"List of {len(recommended_movies)} movies recommended for use_id : {user_id} : ")
    for i, (title, genres) in enumerate(recommended_movies, 1):
        logging.info(f"\t {i}. Title: {title}, Genres: {genres}")
    logging.info("*****|*****|*****")
    
    return recommended_movies


##
def parse_args():
    parser = argparse.ArgumentParser(description="Generate movie recommendations for a user using ALS model.")
    parser.add_argument("--user_id", type=int, help="User ID for whom recommendations should be generated", required=True)
    parser.add_argument("--num_recommendations", type=int, help="Number of recommendations desired", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    user_id = args.user_id
    num_recommendations = args.num_recommendations

    # calling the recommender function with passed args
    recommend_movies(user_id, num_recommendations)
