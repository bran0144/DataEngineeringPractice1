# Spark Cluster 
# Master and workers
# RDD - resilient distributed dataset
# Spark dataframe abstraction on top of RDD’s
# Spark Dataframe designed to behave like a SQL database (table with variables in the columns and observations in the rows)

# First, need to create a SparkSession object from the SparkContext
# SparkContext - sort of like the connection to the cluster
# SparkSession - sort of like the interface with the connection

# SparkSession.builder.getOrCreate() - use this to return an existing SparkSession or create a new one if necessary

# SparkSession - has an attribute catalog which lists all the data inside the cluster
# .listTables() - returns the names of all the tables in the cluster

# Can run SQL queries on the tables in the cluster


# Don't change this query
query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = spark.sql(query)

# Show the results
flights10.show()

# To use Dataframes:
# .toPandas()

# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())

# .createDateFrame() - takes a pandas df and returns a Spark DF
# Stored locally, not in the SparkSession catalog
# You can use SparkDF methods, but you can’t access the data in other contexts
# To do that, you need to create a temporary table
# .createTempView() - on spark df lets you create the temp table - can only be accessed from the specific SparkSession
# .createOrReplaceTempView() - safely creates a temp table or updates an existing table (avoids the problem of duplicate tables)

# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())

# SparkSession has a .read attribute to read text directly into Spark

# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path, header=True)

# Show the data
airports.show()

# .withColumn() - lets you do column wise operations 
# - takes two arguments (string with name of new column, second- new columns itself)
# Spark DF is immutable - columns cannot be updated in place (unlike in pandas)
# df = df.withColumn("newCol", df.oldCol + 1) - overwrites old df

# Create the DataFrame flights
flights = spark.table("flights")

# Show the head
flights.show()

# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time /60) 

# Spark and SQL similarities
# .filter() - similar to WHERE clause

# Both of these work:
# flights.filter("air_time > 120").show()
# flights.filter(flights.air_time > 120).show()

# .filter can take a string
# Or a column of boolean values

# Filter flights by passing a string
long_flights1 = flights.filter("distance > 1000")

# Filter flights by passing a column of boolean values
long_flights2 = flights.filter(flights.distance > 1000)

# Print the data to check they're equal
long_flights1.show()
long_flights2.show()

# .select() is like sql SELECT
# Arguments can be column name as a string or a column object (df.col)
# When you pass a column object, you can do math operations (like add and subtract)
# .select() will return only the columns you want, where .withColumn() returns all the columns (so it is heavier weight). Better to use .select() to not move data you don’t need

# Select the first set of columns
selected1 = flights.select("tailnum", "origin", "dest")

# Select the second set of columns
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter
filterA = flights.origin == "SEA"

# Define second filter
filterB = flights.dest == "PDX"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(filterA).filter(filterB)

# .alias() method allows you to rename the column you’re selecting

# flights.select((flights.air_time/60).alias(“duration_hrs"))

# .selectExpr() - lets you use sql expressions as a string
# flights.selectExpr("air_time/60 as duration_hrs”)

# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed”)
GroupedData methods (.min(), .max(), .count())

# df.groupBy().min(“col”).show()
# created an object, to use the min method, find the min value in col and returns it as a DF

# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Find the longest flight from SEA in terms of air time
flights.filter(flights.origin == “SEA").groupBy().max("air_time").show()

# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum(“duration_hrs").show()

# Grouped dataFrames: 
# pyspark.sql.GroupedData

# Can pass one or more columns to the .groupBy() , just like in SQL

# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count().show()

# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg(“air_time").show()

# .agg() method - can pass an aggregate column expression that uses functions from pyspark.sql.functions

# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation of departure delay
by_month_dest.agg(F.stddev(“dep_delay")).show()

# Joins
# .join() - takes three arguments
# - second df that you want to join
# - on - name of key column as a string (names of key column must be same in each table
# - how - specifies the kind of join - common is “leftouter”  

# Examine the data
print(airports.show())

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

# Examine the new DataFrame
print(flights_with_airports.show())

# Machine learning pipelines
# pyspark.ml library - two main classes- Transformer, Estimator
# Transformer - transform() - takes a df and returns a new df (usually the original with a new column) - other common classes are Bucketizer and PCA
# Estimator - all implement the fit() method. Takes a df and returns a model object (like StringIndexerModel or RandomForestsModel)

# Rename year column
planes = planes.withColumnRenamed("year", "plane_year")

# Join the DataFrames
model_data = flights.join(planes, on="tailnum", how="leftouter")

# Spark only handles numeric data, so types must be ints or floats (called doubles in spark)
# To tell spark how to import datatypes: 
# .cast() method with the withColumn() method
# .withColumn(cast(“integer”))

# Cast the columns to integers
model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))
model_data = model_data.withColumn("air_time", model_data.air_time.cast("integer"))
model_data = model_data.withColumn("month", model_data.month.cast("integer"))
model_data = model_data.withColumn("plane_year", model_data.plane_year.cast(“integer"))

# Create the column plane_age
model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)

# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer"))

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL”)

# How to deal with strings, when spark wants numeric data?
# pyspark.ml.features submodule
# Can create ‘one-hot vectors’
# Steps:
# - create a StringIndexer to encode categorical data (Members of this class are Estimators that take a DataFrame with a column of strings and map each unique string to a number. Then, the Estimator returns a Transformer that takes a DataFrame, attaches the mapping to it as metadata, and returns a new DataFrame with a numeric column corresponding to the string column.
# - use OneHotEncoder - creates an Estimator, then a Transformer (encodes your feature as a vector for ml models

# Create a StringIndexer
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")

# Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol="carrier_index", outputCol=“carrier_fact")

# Create a StringIndexer
dest_indexer = StringIndexer(inputCol="dest", outputCol="dest_index")

# Create a OneHotEncoder
dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol=“dest_fact")

# Last step in the pipeline is to combine all the columns containing features into a single column for ml
# pyspark.ml-feature has a VectorAssembler class for this

# Make a VectorAssembler
vec_assembler = VectorAssembler(inputCols=["month", "air_time", "carrier_fact", "dest_fact", "plane_age"], outputCol=“features")

# Import Pipeline
from pyspark.ml import Pipeline

# Make the pipeline
flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])

# Make sure to only split data into test/train after all of the transformations

# Fit and transform the data
piped_data = flights_pipe.fit(model_data).transform(model_data)

# Split the data into training and test sets
training, test = piped_data.randomSplit([.6, .4])

# logistic regression (predicts probability instead of a numeric variable (like in linear regression))
# Need to classify a cut off point (above is a "yes", below is a "no")
# Hyperparameters - a value in the data that is not esitmated from the data (it is suplied by the user to 
# maximize performance)

# Import LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Create a LogisticRegression Estimator
lr = LogisticRegression()

# Cross validiation
# k-fold - method that estimates the model's performance on unseen data
# splits the training data into different partitions (default is 3).One partition is set aside
# and the model is training with the other data. Error is measured against held out data. Repeated
# with other partitions, until every partition is held out and used as a test exactly once. The error 
# on each of the partitions is averaged (called cross validation error).
# two parameters for this exercise: elasticNetParam, regParam
# Need to commpare different models- BinaryClassificationEvaluator (calculates the area under the ROC)
# Combines the two kind of errors the binary classifier can make (false positives and false negatives)
# into a simple number

# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

# ParamGridBuilder - builds a grid of values to search over to find optimal hyperparameters

# Import the tuning submodule
import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0,1])

# Build the grid
grid = grid.build()

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )

# Fit cross validation models
models = cv.fit(training)

# Extract the best model
best_lr = models.bestModel

# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

# AUC- common metric for binary classification algorithms (area under curve) (closer to 1)
# ROC- recieving operating curve

# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))