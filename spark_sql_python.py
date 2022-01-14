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

