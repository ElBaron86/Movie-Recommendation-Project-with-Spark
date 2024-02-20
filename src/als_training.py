"""\
this script contains functions to build and use a recommender system based on ALS algorithm.

"""

# Modules importation
import time
import logging
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Modules for ALS model
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# Logging configuration
logging.basicConfig(filename="logs/als_training.log", level=logging.INFO,
                    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")

# Configuration of a Spark session 
spark = SparkSession.builder.appName("ALS").getOrCreate()

# Creation of a Spark Context object
sc = spark.sparkContext 

# Importing cleaning function from data_cleaning.py
import sys
sys.path.append("..")

from data_cleaning import cleaning

data = cleaning()

logging.info("Data correctly loaded.")

# Creating en temp view to make sql queriy easly 
data.createOrReplaceTempView("data")

# Data filtering to retain only movies with more than 100 notes 
data_relevant = spark.sql("""
    SELECT *
    FROM data
    WHERE movieId IN (
        SELECT movieId
        FROM data
        GROUP BY movieId
        HAVING COUNT(rating) > 100
    )
""")

# Resize the data to prevent OutOfMemory error
data_relevant = data_relevant.limit(1000000)

# Splitting the data into train and test splits
train_ratio, test_ratio = 0.8, 0.2
(train, test) = data_relevant.randomSplit([train_ratio, test_ratio])

logging.info(f"Data splited in train (with ratio {train_ratio}) and test (with ratio {test_ratio})")

logging.info("### ALS model training... ###")

start_time = time.time()
# Construction of the collaborative filtering model
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

# Definition of the hyperparameter grid
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 15, 20]) \
    .addGrid(als.maxIter, [5, 10, 15]) \
    .addGrid(als.regParam, [0.1, 0.3, 0.5]) \
    .build()

# Initialization of the Trainvalidationsplit
tvs = TrainValidationSplit(estimator=als,
                           estimatorParamMaps=param_grid,
                           evaluator=RegressionEvaluator(metricName="rmse", labelCol="rating"),
                           trainRatio=0.8)  # 80% of data will be used for training

# Model training
model = tvs.fit(train)

end_time = time.time()

logging.info(f"### Training finished. Took {end_time - start_time} seconds ###")

logging.info("### Evaluating in test split ###")

# Model assessment on the test set
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
logging.info(f"### Root Mean Squared Error (RMSE) on test data = {rmse} ###")

# We recover the best model from grid searsh
best_model = model.bestModel

# Saving the model
try:
    best_model.save("src/als_model")
    logging.info("Model successfully saved !")
except Exception as e:
    logging.error(f"Error during model saving : {e}")
