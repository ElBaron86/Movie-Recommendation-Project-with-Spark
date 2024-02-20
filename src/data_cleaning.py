"""\
this script contains basic treatments for cleaning and joining the bases of MovieLens.

"""

# Modules importation

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Configuration of a Spark session 
spark = SparkSession.builder.appName("PremiersPas").getOrCreate()

# Creation of a Spark Context object
sc = spark.sparkContext 

def cleaning():
    """
    This function performs basic treatments for cleaning and joining the bases of MovieLens.

    Returns:
    --------
        DataFrame: A cleaned and joined DataFrame containing movie and rating data.

    """
    

    # movies file
    movies = spark.read.csv('data/ml-25m/movies.csv',
                            header=True,       
                            quote='"',         
                            sep=",",           
                            inferSchema=True,
                            encoding='UTF-8') 

    movies = movies.na.drop(subset=["title"]) # drop movies without title

    # ratings
    ratings = spark.read.csv('data/ml-25m/ratings.csv',
                            header=True,       
                            quote='"',         
                            sep=",",           
                            inferSchema=True,
                            encoding='UTF-8') 

    # tags 
    tags = spark.read.csv('data/ml-25m/tags.csv',
                        header=True,
                        quote = '"',
                        sep = ',',
                        inferSchema=True,
                        encoding='UTF-8')

    # Joining movies and ratings dataframes
    data = movies.join(ratings, on='movieId', how='inner')

    # Deleting movies and ratings to get memory free
    del movies, ratings, tags

    # Transforming timestamp to datetime
    data = data.withColumn("date", from_unixtime("timestamp")).drop("timestamp")

    # Removing movies with non genres listed
    data = data.filter(col('genres') != "(no genres listed)")

    return data
