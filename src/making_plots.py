"""\
this script contains tools to generate the plots used in the README.

"""

# Modules importation
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions as F

# Logging configuration
logging.basicConfig(filename="logs/plots.log", level=logging.INFO,
                    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")

# Plots configuration
custom = {"axes.edgecolor": "purple", "grid.linestyle": "dashed", "grid.color": "black"}
sns.set_style("darkgrid", rc = custom)

# Configuration of a Spark session 
spark = SparkSession.builder.appName("My session").getOrCreate()

# Creation of a Spark Context object
sc = spark.sparkContext 

# Warnings configuration
import warnings
warnings.filterwarnings('ignore')
spark.sparkContext.setLogLevel("ERROR")

# Importing cleaning function from data_cleaning.py
import sys
sys.path.append("..")

from data_cleaning import cleaning

data = cleaning()

#### Movies genres diagram
logging.info('Generating movies genres diagram')

genres = data.select(["genres"])
genres = genres.withColumn('genre', explode(split(col('genres'), '\\|'))).groupby("genre").count().toPandas()

plt.figure(figsize=(6, 4), dpi=130)
plt.title("Movies genres diagram", color="purple")
sns.barplot(genres, x='count', y='genre')
plt.xlabel("nb votes")
plt.legend()
plt.tight_layout()
plt.savefig("images/genres_diagram.png")

# Deleting temp dataframe
del genres

#### Ratings Histogram
logging.info('Generating ratings histogram')

# Creating en temp view to make sql queriy easly 
data.createOrReplaceTempView("view_data")

# SQL query
ratings = spark.sql("SELECT rating, COUNT(*) as count FROM view_data GROUP BY rating")

# Converting to pandas and sorting values
ratings = ratings.toPandas()
ratings.sort_values('rating', ascending=True, inplace=True)

# We color according to the rating slices
colors = ['orange' if x < 2.5 else 'yellow' if 2.5 <= x <= 3.5 else 'blue' for x in ratings['rating']]

# Making the plot
plt.figure(figsize=(6, 4), dpi=130)
sns.barplot(data=ratings, x="rating", y='count', palette=colors)
plt.title("Histogram of movies ratings", color="purple")
plt.savefig("images/notes_bar.png")

# deleting ratings
del ratings

#### Movies ranking diagram
logging.info('Generating movies ranking diagram')

data.createOrReplaceTempView("movie_ratings")

# Calculating the ratings average and transform to pandas
average_ratings = spark.sql("""
    SELECT movieId, title, genres, AVG(rating) AS average_rating, COUNT(rating) AS n_rates
    FROM movie_ratings
    GROUP BY movieId, title, genres
""").toPandas().sort_values('average_rating', ascending=False)

plt.figure(figsize=(7, 5), dpi=150)
plt.title("Ranking of rated movies", color="purple")
custom_palette = sns.color_palette(palette='crest')

# Filtering movies with more than 100 votes and taking the first 10
filtered_data = average_ratings[average_ratings["n_rates"] > 100].head(10)

# Diagram
sns.barplot(data=filtered_data, y="title", x="average_rating", hue="n_rates", orient="h", palette='crest')

plt.xlabel('Average Rating')
plt.ylabel('Movie Title')
plt.legend(title='Number of Ratings', bbox_to_anchor=(1.0, 1), loc='upper left')
plt.tight_layout()
plt.savefig("images/movies_ranging.png")

# deleting temp dataframes
del filtered_data, average_ratings


#### Series of movies genres average rate
logging.info('Generating series of movies genres average rate')


# Dataframe with essential columns
genres_ranking_series = data.select(['genres', 'rating', 'date'])

# Genres separation (some movies have several genres)
genres_ranking_series = genres_ranking_series.withColumn('genre', explode(split(col('genres'), '\\|')))

# Extracting the year from the 'date' column
genres_ranking_series = genres_ranking_series.withColumn('year', year(col('date')))

# Calculating the average note per gender and per year
genres_ranking_series = genres_ranking_series.groupBy('genre', 'year').agg({'rating': 'avg'}).withColumnRenamed('avg(rating)', 'average_rating')

# Import suplementary classes and functions
from pyspark.sql.window import Window
from pyspark.sql.functions import rank

# Definition of a window specification
window_spec = (
    Window.partitionBy('year')   # Partition the data by year
          .orderBy('average_rating')  # Order the data by average rating
)

# Assigning ranks to records within each partition of the window
ranked_genres = (
    genres_ranking_series         # DataFrame to apply rank to
    .withColumn('rank', rank().over(window_spec))  # Add a 'rank' column with calculated ranks
)


# Selecting top 5 genres per year
genres_ranking_series = ranked_genres.filter(col('rank') <= 5).orderBy('year', 'rank')


# Converting to pandas
genres_ranking_series = genres_ranking_series.toPandas()

# Generate the plot
plt.figure(figsize=(12, 8), dpi=220)
sns.lineplot(data=genres_ranking_series, x='year', y='average_rating', hue='genre', marker='o')
plt.title("Series of movies genre avg rate", color="purple")
plt.xlabel("Year")
plt.ylabel("Average rate")
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("images/genres_ranging_series.png")

# deleting temp dataframe
del genres_ranking_series

#### Scatterplot of avg rate vs votes count
logging.info('Generating avg rate vs n_votes')

# Calculating average rating and number of votes for each movie
average_ratings = data.groupBy('movieId').agg({'rating': 'mean', 'userId': 'count'}).withColumnRenamed('avg(rating)', 'average_rating').withColumnRenamed('count(userId)', 'n_votes')

# Convert to Pandas DataFrame
average_ratings = average_ratings.toPandas()

# Scatter plot
plt.figure(figsize=(5, 5), dpi=150)
plt.scatter(average_ratings['n_votes'], average_ratings['average_rating'], color='purple', alpha=0.5)
plt.title('Average rating vs number of votes')
plt.xlabel('Number of votes')
plt.ylabel('avg rating')
plt.tight_layout()
plt.savefig("images/rating_vs_votes.png")
