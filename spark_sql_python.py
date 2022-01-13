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

