# What is data cleaning?
# preparing raw data for use in data processing pipelines
# Necessary for any production level data system
# Possible tasks:
    # reformatting or replacing text
    # performing calculations
    # removing garbage or imcomplete data
# Problems with typical data systems for cleaning data:
    # performance
    # organizing data flow
# Spark advantages:
    # scalable
    # powerful framework for data handling  
    # built-in ability to validate data with schema   
# Schemas 
    # define the format of a DF
    # may contain various types (strings, int, etc)
    # can filter garbage data during import
    # improve read performance
    # defining a schema and checking on import means you only need to do read once
import pyspark.sql.types
peopleSchema = StructType([
    StructField('name', StringType(), True),
    StructField('age', IntegerType(), True),
    StructField('city', StringType(), True)
]) 

people_df = spark.read.format('csv').load(name='rawdata.csv', schema=peopleSchema)
