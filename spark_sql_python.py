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

