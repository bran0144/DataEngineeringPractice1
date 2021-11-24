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

