#To load a DF from a file:
df = spark.read.csv('filename', header=True)
#To create SQL table and query it:
df.createOrReplaceTempView("schedule")  #schedule is the desired name of the table
spark.sql("SELECT * FROM schedule WHERE station = 'San Jose'").show()
#To inspect the table schema:
result = spark.sql("SHOW COLUMNS FROM tablename")
#Another way
result = spark.sql("SELECT * FROM tablename LIMIT 0")
#Another way
result = spark.sql("DESCRIBE tablename")
result.show()
print(result.columns)
#SQL table in spark lets you treat the distributed data like one large db

#Exercises:
# Load trainsched.txt
df = spark.read.csv("trainsched.txt", header=True)

# Create temporary table called table1
df.createOrReplaceTempView("table1")

# Inspect the columns in the table df
spark.sql("DESCRIBE schedule").show()

#Window FunctionSQL
    #simpler notation that dot notation or queries
    #each row uses the values of other rows to calculate its value
    #like the difference between values (time to next stop)
query = """
SELECT train_id, station, time,
LEAD(time, 1) OVER (ORDERBY time) AS time_next
FROM sched
WHERE train_id=324 """  
    #OVER clause is designated as a window clause
    #OVER clause must contain as ORDER BY clause
    #LEAD lets you query more than one row at a time without having to join table to itself
query = """
SELECT train_id, station, time,
LEAD(time, 1) OVER (PARTITION BY train_id ORDERBY time) AS time_next
FROM sched """  
    #returns more train id's, but groups them together

#Exercises:
# Add col running_total that sums diff_min col in each group
query = """
SELECT train_id, station, time, diff_min,
SUM(diff_min) OVER (PARTITION BY train_id ORDER BY time) AS running_total
FROM schedule
"""

# Run the query and display the result
spark.sql(query).show()

#Dot notation and SQL
#selects 2 columns
df.select('train_id', 'station').show(5)
#using dot notation
df.select(df.train_id, df.station)
#using col
from _typeshed import FileDescriptor
from pyspark.sql.functions import col
df.select(col('train_id'), col('station'))
#to rename a column
df.select('train_id', 'station').withColumnRenamed('train_id', 'train')
#or this
df.select(col('train_id').alias('train'), 'station')
#using SQL
spark.sql('SELECT train_id AS train, station FROM schedule LIMIT 5').show()
#or like this
df.select(col('train_id').alias('train'), 'station').limit(5).show()
#Window function in SQL
query = """
SELECT *,
ROW_NUMBER() OVER (PARTITION BY train_id ORDER BY time) AS id
FROM schedule"""
spark.sql(query).show(11)
#using dot notation
from pyspark.sql import Window
from pyspark.sql.functions import row_number
df.withColumn("id", row_number().over(
    Window.partitionBy('train_id')
    .orderBy('time')
))
#Using a WindowSpec
#over function in SparkSQL corresponds to OVER clause in SQL
window = Window.partitionBy('train_id').orderBy('time')
dfx = df.withColumn('next', lead('time', 1).over(window))

#Exercises:
# Give the identical result in each command
spark.sql('SELECT train_id, MIN(time) AS start FROM schedule GROUP BY train_id').show()
df.groupBy('train_id').agg({'time':'min'}).withColumnRenamed('min(time)', 'start').show()

# Print the second column of the result
spark.sql('SELECT train_id, MIN(time), MAX(time) FROM schedule GROUP BY train_id').show()
result = df.groupBy('train_id').agg({'time':'min', 'time':'max'})
result.show()
print(result.columns[1])

# Write a SQL query giving a result identical to dot_df
query = "SELECT train_id, MIN(time) AS start, MAX(time) AS end FROM schedule GROUP BY train_id"
sql_df = spark.sql(query)
sql_df.show()

df = spark.sql("""
SELECT *, 
LEAD(time,1) OVER(PARTITION BY train_id ORDER BY time) AS time_next 
FROM schedule
""")
# Obtain the identical result using dot notation 
dot_df = df.withColumn('time_next', lead('time', 1)
        .over(Window.partitionBy('train_id')
        .orderBy('time')))

#dot notation
window = Window.partitionBy('train_id').orderBy('time')
dot_df = df.withColumn('diff_min', 
                    (unix_timestamp(lead('time', 1).over(window),'H:m') 
                     - unix_timestamp('time', 'H:m'))/60)
# Create a SQL query to obtain an identical result to dot_df
query = """
SELECT *, 
(UNIX_TIMESTAMP(LEAD(time, 1) OVER (PARTITION BY train_id ORDER BY time),'H:m') 
 - UNIX_TIMESTAMP(time, 'H:m'))/60 AS diff_min 
FROM schedule 
"""
sql_df = spark.sql(query)
sql_df.show()

#using natural language processing
df = spark.read.text('sherlock.txt')
print(df.first())
df.count()
#loading parquet
df1 = spark.read.load('sherlock.parquet')
df1.show(15, truncate=False)
#convert to lower case
df = df1.select(lower(col('value')))
#alias columns
df = df1.select(lower(col('value')).alis('v'))
#replacing text
df = df1.select(regexp_replace('value', "Mr\.", 'Mr').alias('v'))
df = df2.select(regexp_replace('value', 'don\'t', 'do not').alias('v'))
#tokenizing text
df = df2.select(split('v', '[ ]').alias('words'))
df.show(truncate=False)
puncuation = "_|.\?\!\",\'\[\]\*()"
df3 = df2.select(split('v', '[ %s]' % puncuation).alias('words'))
#exploding an array (increases row count)
df4 = df3.select(explode('words').alias('word'))
#remove empty rows
nonblack_df = df.where(length('word') > 0)
#adding a row id column
df2 = df.select('word', monotonically_increasing_id().alias('id'))
#partitioning the data
df2 = df.withColumn('title', when(df.id < 25000, 'Preface')
.when(df.id < 50000, 'Chapter 1')
.when(df.id < 75000, 'Chapter 2')
.otherwise('Chapter 3'))
#repartionting on a column
df2 = df.repartition(4, 'part')
print(df2.rdd.getNumPartitions())
#reading pre-partitioned text - reads files in parallel
df_parts = spark.read.text('sherlock_parts')

#Exercises:
# Load the dataframe
df = spark.read.load('sherlock_sentences.parquet')

# Filter and show the first 5 rows
df.where('id > 70').show(5, truncate=False)

# Split the clause column into a column called words 
split_df = clauses_df.select(split('clause', ' ').alias('words'))
split_df.show(5, truncate=False)

# Explode the words column into a column called word 
exploded_df = split_df.select(explode('words').alias('word'))
exploded_df.show(10)

# Count the resulting number of rows in exploded_df
print("\nNumber of rows: ", exploded_df.count())

#Moving window analysis
df.select('part', 'title').distinct().sort('part').show(truncate=False)

#Partitions
df.select('part', 'title'),distinct().sort('part').show(truncate=False)
#moving windwo analysis (selects some of the data)
query = """
SELECT id, word AS w1,
LEAD(word,1) OVER(PARTITION BY part ORDER BY id) AS w2,
LEAD(word,2) OVER(PARTITION BY part ORDER BY id) AS w3
FROM df"""
spark.sql(query).sort('id').show()
#moving windows are useful for algorithms (predictive text?)
#LAG winow function
lag_query="""
SELCT id,
LAG(word,2) OVER(PARTITION BY part ORDER BY id) AS w1,
LAG(word,1) OVER(PARTITION BY part ORDER BY id) AS w2,
word AS w3
FROM df
ORDR BY id  """
spark.sql(lag_query).show()
#starts with a bunch of null values
#here windows stay within partition
lag_query="""
SELCT id,
LAG(word,2) OVER(PARTITION BY part ORDER BY id) AS w1,
LAG(word,1) OVER(PARTITION BY part ORDER BY id) AS w2,
word AS w3
FROM df
WHERE part=2 """
spark.sql(lag_query).show()

#Exercises:
# Word for each row, previous two and subsequent two words
query = """
SELECT
part,
LAG(word, 2) OVER(PARTITION BY part ORDER BY id) AS w1,
LAG(word, 1) OVER(PARTITION BY part ORDER BY id) AS w2,
word AS w3,
LEAD(word, 1) OVER(PARTITION BY part ORDER BY id) AS w4,
LEAD(word, 2) OVER(PARTITION BY part ORDER BY id) AS w5
FROM text
"""
spark.sql(query).where("part = 12").show(10)

# Repartition text_df into 12 partitions on 'chapter' column
repart_df = text_df.repartition(12, 'chapter')

# Prove that repart_df has 12 partitions
repart_df.rdd.getNumPartitions()

#Common word sequences (for predictive text)
    #words represented by symbolic tokens
    #can also predict/recommend song or video based on previous views
    #categorical data(nominal/qualitative)
    #tend not to have a logical order
    #if they have order , then they are ordinal (like a rating)
#Sequence analysis
    #look at words that tend to appear together
#3tuples
query3 = """
SELECT id, word as w1,
LEAD(word,1) OVER(PARTITION BY part ORDER BY id) AS w2,
LEAD(word,2) OVER(PARTITION BY part ORDER BY id) AS w3
FROM df"""

query3agg = """
SELECT w1,w2,w3, COUNT(*) as count FROM (
    SELECT word as w1,
LEAD(word,1) OVER(PARTITION BY part ORDER BY id) AS w2,
LEAD(word,2) OVER(PARTITION BY part ORDER BY id) AS w3
FROM df)
GROUP BY w1,w2,w3
ORDER BY count DESC"""
spark.sql(query3agg).show()
#returns the count of each 3 tuple (gives the most common 3tuples)

query4agg = """
SELECT w1,w2,w3, length(w1)+length(w2)+lenth(w3) as length FROM (
    SELECT word as w1,
LEAD(word,1) OVER(PARTITION BY part ORDER BY id) AS w2,
LEAD(word,2) OVER(PARTITION BY part ORDER BY id) AS w3
FROM df)
GROUP BY w1,w2,w3
ORDER BY length DESC"""
spark.sql(query4agg).show(truncate=False)
#returns the longest 3tuples

#Exercises:
# Find the top 10 sequences of five words
query = """
SELECT w1, w2, w3, w4, w5, COUNT(*) AS count FROM (
   SELECT word AS w1,
   LEAD(word,1) OVER(PARTITION BY part ORDER by id) AS w2,
   LEAD(word,2) OVER(PARTITION BY part ORDER by id) AS w3,
   LEAD(word,3) OVER(PARTITION BY part ORDER by id) AS w4,
   LEAD(word,4) OVER(PARTITION BY part ORDER by id) AS w5
   FROM text
)
GROUP BY w1, w2, w3, w4, w5
ORDER BY count DESC
LIMIT 10 """
df = spark.sql(query)
df.show()

# Unique 5-tuples sorted in descending order
query = """
SELECT DISTINCT w1, w2, w3, w4, w5 FROM (
   SELECT word AS w1,
   LEAD(word,1) OVER(PARTITION BY part ORDER BY id ) AS w2,
   LEAD(word,2) OVER(PARTITION BY part ORDER BY id ) AS w3,
   LEAD(word,3) OVER(PARTITION BY part ORDER BY id ) AS w4,
   LEAD(word,4) OVER(PARTITION BY part ORDER BY id ) AS w5
   FROM text
)
ORDER BY w1 DESC, w2 DESC, w3 DESC, w4 DESC, w5 DESC
LIMIT 10
"""
df = spark.sql(query)
df.show()

#   Most frequent 3-tuple per chapter
query = """
SELECT chapter, w1, w2, w3, count FROM
(
  SELECT
  chapter,
  ROW_NUMBER() OVER (PARTITION BY chapter ORDER BY count DESC) AS row,
  w1, w2, w3, count
  FROM ( %s )
)
WHERE row = 1
ORDER BY chapter ASC
""" % subquery

spark.sql(query).show()

#Caching
#Spark tends to unload data aggressively
#Eviction Policy
    #LRU - Least Recently Used - default policy
    #eviction happens independently on each worker
    #depends on memory available to each worker
#To cache a df:
df.cache()
#To uncache:
df.unpersist()
#To determine if a df is cached:
df.is_cached()

#Storage Level
    #specifies 5 details about how it is cached
df.storageLevel
#output
StorageLevel(True, True, False, True, 1)
    #useDisk, useMemory, useOffHeap, deserialized, replicaton
#useDisk - specifies whether to move some or all of df to disk if needed to free up memory
#use Memory - specifies whether to keep data in memory
#useOffHeap- tells Spark to use OffHeap instead of onHeap memory
#deserialized - specifies whether data being stored is serialized(takes up less space, but slower to load)
    #only applies to in memory storage
    #DiskCache is always serialized
#replication - used to tell Spark to replicate data on multiple nodes
df.persist(storageLevel=pyspark.StorageLevel.MEMORY_AND_DISK)
    #lets you specify 
#Caching a table
df.createOrReplaceTempView('df')
spark.catalog.isCached(tableName='df')
#then to cache it:
spark.catalog.cacheTable('df')
#to uncache:
spark.catalog.uncacheTable('df')
#to remove cache:
spark.catalog.clearCache()
#Tips
    #caching is a lazy operation - won't appear in cache until action is performed on df
    #Only cache if more than one operation is to be performed
    #unpersist when you no longer need the object
    #cache selectively - it can really slow things down

#Exercises:
# Unpersists df1 and df2 and initializes a timer
prep(df1, df2) 

# Cache df1
df1.cache()

# Run actions on both dataframes
run(df1, "df1_1st") 
run(df1, "df1_2nd")
run(df2, "df2_1st")
run(df2, "df2_2nd", elapsed=True)

# Prove df1 is cached
print(df1.is_cached)

# Unpersist df1 and df2 and initializes a timer
prep(df1, df2) 

# Persist df2 using memory and disk storage level 
df2.persist(storageLevel=pyspark.StorageLevel.MEMORY_AND_DISK)

# Run actions both dataframes
run(df1, "df1_1st") 
run(df1, "df1_2nd") 
run(df2, "df2_1st") 
run(df2, "df2_2nd", elapsed=True)

# List the tables
print("Tables:\n", spark.catalog.listTables())

# Cache table1 and Confirm that it is cached
spark.catalog.cacheTable('table1')
print("table1 is cached: ", spark.catalog.isCached('table1'))

# Uncache table1 and confirm that it is uncached
spark.catalog.uncacheTable('table1')
print("table1 is cached: ", spark.catalog.isCached('table1'))

#Spark UI
# Spark Task - unit of execution that runs on a single cpu
# Spark Stage - group of tasks that perform the same computation in parallel , typically on a different 
# subset of the data
# Spark Job - computation triggered by an action, sliced into one or more stages
# Will be found at localhost:4040 (if in use, try 4041, 4042, 4043)
# 6 tabs - jobs, stages, storage, environment, executors, SQL
#spark.catalog.dropTempView(‘table1’) - removes temporary table from catalog
#Spark Catalog - 
# spark.catalog.listTables() - lists tables that exist and their properties
# UI Storage Tab - shows where data partitions exist (in memory, on disk, across a cluster, at the snapshot in time)
# Stages are presented in reverse chronological order

# # Logging
# import logging
# logging.basicConfig(stream=sys.std.out, level=logging.INFO, 
# format=‘%(asctime)s - %(levelname)s - %(message)s’)
# logging.info(“Hello %s, “world”)
# logging.debug(“Hello, take %d”, 2)

# Debugging with lazy evaluation and distributed execution can be challenging
# Using a timer to prove stealth loss of CPU
t = timer()
t.elasped()
t.reset()
t.elapsed()

class timer:
    start_time = time.time()
    step = 0

def elapsed(self, reset=True):
    self.step += 1
    print(“%d. Elapsed: %.1f sec %s”
    % (self.step, time.time() - self.start_time))
    if reset:
        self.reset()
def reset(self):
    self.start_time = time.time()

# import logging
# logging.basicConfig(stream=sys.std.out, level=logging.INFO, 
# format=‘%(asctime)s - %(levelname)s - %(message)s’)
# t = timer()
# logger.info("NO action here.”)
# t.elapsed()
# logging.debug(“df has %d rows.” df.count())
# t.elapsed()

# Disable actions to prevent loss
# ENABLED = False
# t = timer()
# logging.info("NO action here.”)
# t.elapsed()
# If ENABLED:
# logging.info(“df has %d rows.” df.count())
# t.elapsed()

# Exercises:

# Log columns of text_df as debug message
logging.debug("text_df columns: %s", text_df.columns)

# Log whether table1 is cached as info message
logging.info("table1 is cached: %s", spark.catalog.isCached(tableName="table1"))

# Log first row of text_df as warning message
logging.warning("The first row of text_df:\n %s", text_df.first())

# Log selected columns of text_df as error message
logging.error("Selected columns: %s", text_df.select("id", “word"))

# Uncomment the 5 statements that do NOT trigger text_df
logging.debug("text_df columns: %s", text_df.columns)
logging.info("table1 is cached: %s", spark.catalog.isCached(tableName="table1"))
# logging.warning("The first row of text_df: %s", text_df.first())
logging.error("Selected columns: %s", text_df.select("id", "word"))
logging.info("Tables: %s", spark.sql("show tables").collect())
logging.debug("First row: %s", spark.sql("SELECT * FROM table1 limit 1"))
# logging.debug("Count: %s", spark.sql("SELECT COUNT(*) AS count FROM table1”).collect())

# EXPLAIN SELECT * FROM table1
# Returns a query plan 
df = sqlContext.read.load(‘/temp/df.parquet’)
df.registerTempTable(‘df’)
spark.sql(‘EXPLAIN SELECT * FROM df’).first()
#Can also run 
df.explain() on df’s

spark.sql(“SELECT * FROM df”).explain()

# Words sorted by frequency query
# SELECT word, COUNT(*) AS count
# FROM df
# GROUP BY word
# ORDER BY count DESC

df.groupBy(‘word’).count().sort(desc(‘count’)).explain()

# Exercises:

# Run explain on text_df
text_df.explain()

# Run explain on "SELECT COUNT(*) AS count FROM table1" 
spark.sql("SELECT COUNT(*) AS count FROM table1").explain()

# Run explain on "SELECT COUNT(DISTINCT word) AS words FROM table1"
spark.sql("SELECT COUNT(DISTINCT word) AS words FROM table1”).explain()

# Extract Transform Select

# Extraction involves extracting features from raw data. Transformation involves scaling, converting, or 
# modifying features. Selection obtains a subset of features.

From pyspark.sql.functions import split, explode, length
df.where(length(‘sentence’) == 0)
UDF - user defined function

From pyspark.sql.functions import udf
From pyspark.sql.types import BooleanType
short_udf = udf(lambda x: True if not x or len(x) < 10 else False, BooleanType())

df.select(short_udf(‘testdata’).alias(‘is_short)).show(3)

From pyspark.sql.types import StringType, IntegerType, FloatType, ArrayType
df3.select(‘word array’, in_udf(‘word array’).alias(‘without endword)).show(5, truncate=30)

in_udf = udf(lambda x: x[0:len(x)-1] if x and len(x) > 1
else [],
ArrayType(StringType()))

# Sparse vector format
# Indices
# Values
# Array example [1.0, 0.0, 0.0, 3.0]
# Sparse vector example (4, [0,3], [1.0, 3.0])

# Working with vector data
# hasattr(x, “toArray”)
# x.numNonzeros())

# Returns true if the value is a nonempty vector
nonempty_udf = udf(lambda x:  
    True if (x and hasattr(x, "toArray") and x.numNonzeros())
    else False, BooleanType())

# Returns first element of the array as string
s_udf = udf(lambda x: str(x[0]) if (x and type(x) is list and len(x) > 0)
    else '', StringType())

# UDF removes items in TRIVIAL_TOKENS from array
rm_trivial_udf = udf(lambda x:
                     list(set(x) - TRIVIAL_TOKENS) if x
                     else x,
                     ArrayType(StringType()))

# Remove trivial tokens from 'in' and 'out' columns of df2
df_after = df_before.withColumn('in', rm_trivial_udf('in'))\
                    .withColumn('out', rm_trivial_udf('out'))

# Show the rows of df_after where doc contains the item '5'
df_after.where(array_contains('doc','5')).show()

#Feature data for classification
#creating a udf that gets first element of array and converts it to an int
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
bad_udf = udf(lambda x:
    x.indices[0]
    if (x and hasattr(x, "toArray") and x.numNonzeros())
    else 0,
    IntegerType())

try:
    df.select(bad_udf('outvec').alias('label')).first()
except Exception as e:
    print(e.__class__)
    print(e.errmsg)

# Need to cast into an int first
first_udf = udf(lambda x:
    int(x.indices[0])
    if (x and hasattr(x, "toArray") and x.numNonzeros())
    else 0,
    IntegerType())

df.withColumn('label', k_udf('outvec')).drop('outvec').show(3)

# Count vectorizer
    #feature extractor
    #its input is an array of strings
    #output is a vector
from spark.ml.feature improt CountVectorizer
cv = CountVectorizer(inputCol='words', outputCol='features')
model = cv.fit(df)
result = model.transform(df)
print(result)

#Exercises
# Selects the first element of a vector column
first_udf = udf(lambda x:
            float(x.indices[0]) 
            if (x and hasattr(x, "toArray") and x.numNonzeros())
            else 0.0,
            FloatType())

# Apply first_udf to the output column
df.select(first_udf("output").alias("result")).show(5)

# Add label by applying the get_first_udf to output column
df_new = df.withColumn('label', get_first_udf('output'))

# Show the first five rows 
df_new.show(5)

# Transform df using model
result = model.transform(df.withColumnRenamed('in', 'words'))\
        .withColumnRenamed('words', 'in')\
        .withColumnRenamed('vec', 'invec')
result.drop('sentence').show(3, False)

# Add a column based on the out column called outvec
result = model.transform(result.withColumnRenamed('out', 'words'))\
        .withColumnRenamed('words', 'out')\
        .withColumnRenamed('vec', 'outvec')
result.select('invec', 'outvec').show(3, False)	

#Text Classification
#this method does not look at word order for predictive text
#Selecting the data
df_true = df.where("endword in ('she', 'he', 'hers', ' his', 'her', 'him')")
    .withColumn('label', lit(1))
    #will return a 1 if endword is in the set of pronouns listed
df_false = df.where("endword not in ('she', 'he', 'hers', ' his', 'her', 'him')")
    .withColumn('label', lit(0))
#Combine these to df and we have training data
df_examples = df_true.union(df_false)
df_train, df_eval = df_examples.randomSplit((0.60, 0.40), 42)
#Use logistic regression
from pyspark.ml.classification import LogisticRegression
logistic = LogisticRegression(maxIter=50, regParam=0.6, elasticNetParam=0.3)
model  = logistic.fit(df_train)
print("Training iterations: ", model.summary.totalIterations)

#Exercises:
# Import the lit function
from pyspark.sql.functions import lit

# Select the rows where endword is 'him' and label 1
df_pos = df.where("endword = 'him'")\
           .withColumn('label', lit(1))

# Select the rows where endword is not 'him' and label 0
df_neg = df.where("endword <> 'him'")\
           .withColumn('label', lit(0))

# Union pos and neg in equal number
df_examples = df_pos.union(df_neg.limit(df_pos.count()))
print("Number of examples: ", df_examples.count())
df_examples.where("endword <> 'him'").sample(False, .1, 42).show(5)

# Split the examples into train and test, use 80/20 split
df_trainset, df_testset = df_examples.randomSplit((0.80, 0.20), 42)

# Print the number of training examples
print("Number training: ", df_trainset.count())

# Print the number of test examples
print("Number test: ", df_testset.count())

# Import the logistic regression classifier
from pyspark.ml.classification import LogisticRegression

# Instantiate logistic setting elasticnet to 0.0
logistic = LogisticRegression(maxIter=100, regParam=0.4, elasticNetParam=0.0)

# Train the logistic classifer on the trainset
df_fitted = logistic.fit(df_trainset)

# Print the number of training iterations
print("Training iterations: ", df_fitted.summary.totalIterations)

#Predicting and evalutation
predicted = df_trained.transform(df_test)
#prediction column : double
#probability column: vector of length two

x =  predicted.first
print("Right!" if x.label == int(x.prediction) else "Wrong")

#evaluating with AUC
model_stats = model.evaluate(df_eval)
type(model_stats)
print("\nPerformance: %.2f" model_stats.areaUnderROC)

#Exercises:
# Apply the model to the test data
predictions = df_fitted.transform(df_testset).select(fields)

# Print incorrect if prediction does not match label
for x in predictions.take(8):
    print()
    if x.label != int(x.prediction):
        print("INCORRECT ==> ")
    for y in fields:
        print(y,":", x[y])