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

# Immutability and lazy processing
# DF's are immutable (component of functional programming)
    # defined once
    # unable to be directly modified
    # means that you don't need to worry about dealing with concurrent data objects
voter_df = spark.read.csv("voter.csv")
voter_df = voter_df.withColumn("fullyear", voter_df.year + 2000)    #original is destroyed
voter_df = voter_df.drop(voter_df.year)      #original is destroyed

#lazy processing
# little actually happens until an action is performed
# allows efficient planning of operations

#Exercises:
# Load the CSV file
aa_dfw_df = spark.read.format('csv').options(Header=True).load('AA_DFW_2018.csv.gz')

# Add the airport column using the F.lower() method
aa_dfw_df = aa_dfw_df.withColumn('airport', F.lower(aa_dfw_df['Destination Airport']))

# Drop the Destination Airport column
aa_dfw_df = aa_dfw_df.drop(aa_dfw_df['Destination Airport'])

# Show the DataFrame
aa_dfw_df.show()

#Parquet
#Problems with csv:
    #no defined schema
    #nested data requires special handling
    #encoding format is limited
#problems with csv and spark:
    #slow to parse
    # files cannot be filtered prior to processing(no "predicate pushdown")
    # any intermediate use requires redefining schema
#Parquet
    #columnar data format (used in Hadoop, Spark, etc)
    #data is consumed in chunks, allowing accessible read /write 
    #supports predicate pushdown (significant performance improvement)
    #automatically stores schema information and data encoding
    #binary file format
#reading files
df = spark.read.format('parquet').load('filename.parquet')
#OR you can do this:
df= spark.read.parquet('filename.parquet')
#writing files
df.write.format('parquet').save('filename.parquet')
#OR
df.write.parquet('filename.parquet')

#Parquet are great for performing SQL operations
flight_df = spark.read.parquet('flights.parquet')
flight_df.createOrReplaceTempView('flights')    #adds an alias of hte parquet data as a SQL table
short_flights_df = spark.sql('SELECT * FROM flights WHERE flightduration < 100')

#Exercises:
# View the row count of df1 and df2
print("df1 Count: %d" % df1.count())
print("df2 Count: %d" % df2.count())

# Combine the DataFrames into one
df3 = df1.union(df2)

# Save the df3 DataFrame in Parquet format
df3.write.parquet('AA_DFW_ALL.parquet', mode='overwrite')

# Read the Parquet file into a new DataFrame and run a count
print(spark.read.parquet('AA_DFW_ALL.parquet').count())

# Read the Parquet file into flights_df
flights_df = spark.read.parquet('AA_DFW_ALL.parquet')

# Register the temp table
flights_df.createOrReplaceTempView('flights')

# Run a SQL query of the average flight duration
avg_duration = spark.sql('SELECT avg(flight_duration) from flights').collect()[0]
print('The average flight time is: %d' % avg_duration)

#DataFrames in PySpark:
    #made up of rows and columns
    #immutable
    #use transformation operations to modify data
voter_df.filter(voter_df.name.life('M%'))
voters = voter_df.select('name', 'position')

#Common transormations:
    #FIlter/where
voter_df.filter(voter_df.date > '1/1/2019') # OR voter_df.where(...)
    #select - selects columns
voter_df.select(voter_df.name)
    #withColumn - creates a new column
voter_df.withColumn('year', voter_df.date.year)
    #drop - removes a column from the df
voter_df.drop('unused_column')

#Filtering - common in cleaning data:
    #remove nulls
voter_df.filter(voter_df['name'].isNotNull())
    #remove odd entries
voter_df.filter(voter_df.date.year > 1800)
    #split data from combined sources
voter_df.where(voter_df['_c0'].contains('VOTE'))
    #can be negated with ~
voter_df.where( ~ voter_df._c1.isNull())

#Columns string transformations
#contained in pyspark.sql.functions
import pyspark.sql.functions as F
#applied per column as transformation
voter_df.withColumn('upper', F.upper('name'))
#can create intermediary columns only for processing
    #useful for complex transformations requiring many steps
voter_df.withColumn('splits', F.split('name', ' '))
#Casting string data to other types
voter_df.withColumn('year', voter_df['_c4'].cast(IntegerType()))

#Array Type column functions
#.size(<column>) - returns length of array column
#.getItem(<index>) - used to retrieve a specific item at index of list column

# Show the distinct VOTER_NAME entries
voter_df.select('VOTER_NAME').distinct().show(40, truncate=False)

# Filter voter_df where the VOTER_NAME is 1-20 characters in length
voter_df = voter_df.filter('length(VOTER_NAME) > 0 and length(VOTER_NAME) < 20')

# Filter out voter_df where the VOTER_NAME contains an underscore
voter_df = voter_df.filter(~ F.col('VOTER_NAME').contains('_'))

# Show the distinct VOTER_NAME entries again
voter_df.select('VOTER_NAME').distinct().show(40, truncate=False)

# Add a new column called splits separated on whitespace
voter_df = voter_df.withColumn('splits', F.split(voter_df.VOTER_NAME, '\s+'))

# Create a new column called first_name based on the first item in splits
voter_df = voter_df.withColumn('first_name', voter_df.splits.getItem(0))

# Get the last entry of the splits list and create a column called last_name
voter_df = voter_df.withColumn('last_name', voter_df.splits.getItem(F.size('splits') - 1))

# Drop the splits column
voter_df = voter_df.drop('splits')

# Show the voter_df DataFrame
voter_df.show()

#Using Conditionals
#best to use inline conditionals (much better performance)
#.when()
df.select(df.Name, df.Age, F.when(df.Age >= 18, "Adult"))
    #this adds a new column and puts "adult" in the values that evaluate to True (column is unnamed)
df.select(df.Name, df.Age, F.when(df.Age>= 18, "Adult").when(df.Age < 18, "Minor"))
    # can chain .when clauses - I don't think this is correct syntax
#.otherwise() - like else - can only have 1 otherwise, but many when clauses
df.select(df.Name, df.Age, 
    when(df.Age >= 18, "Adult")
    .otherwise("Minor"))

#Exercises
# Add a column to voter_df for any voter with the title **Councilmember**
voter_df = voter_df.withColumn('random_val',
                               when(voter_df.TITLE == 'Councilmember', F.rand()))

# Show some of the DataFrame rows, noting whether the when clause worked
voter_df.show()

# Add a column to voter_df for a voter based on their position
voter_df = voter_df.withColumn('random_val',
                            when(voter_df.TITLE == 'Councilmember', F.rand())
                            .when(voter_df.TITLE == 'Mayor', 2)
                            .otherwise(0))

# Show some of the DataFrame rows
voter_df.show()

# Use the .filter() clause with random_val
voter_df.filter(voter_df.random_val == 0).show()

#User defined functions
#wrapped with pyspark.sql.functions.udf method
#result is stored as a variable
#Reverse string UDF
def reverseString(mystr):
    return mystr[::-1]
#wrap the function and store as a variable
udfReverseString = udf(reverseString, StringType())
user_df = user_df.withColumn('ReverseName', udfReverseString(user_df.Name))
#Argument less example
def sortingCap():
    return random.choice(['G', 'H', 'R', 'S'])
udfSortingCap = udf(sortingCap, StringType())
user_df = user_df.withColumn('Class', udfSortingCap())

#Partitioning and lazy processing
#size can vary, but they should be kept about equal
#each partition is handled independently
#transformations are not run until anaction is performed
#transformations can be reordered for best performance which can sometimes cause
    #unexpected behavior
#ID fields - usually integers increasing, sequential, unique (common in relational db's)
    #can cause problems in parallel processing
#Spark uses monotoically increasing ID's
    #pyspark.sql.functions.monotoically_increasing_id()
    #not necessarily sequentail (gaps exist)
    #integer, unique
    #completely parallel
    #64bit split into groups based on partition
#Exercises:
# Select all the unique council voters
voter_df = df.select(df["VOTER NAME"]).distinct()

# Count the rows in voter_df
print("\nThere are %d rows in the voter_df DataFrame.\n" % voter_df.count())

# Add a ROW_ID
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())

# Show the rows with 10 highest IDs in the set
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)

# Print the number of partitions in each DataFrame
print("\nThere are %d partitions in the voter_df DataFrame.\n" % voter_df.rdd.getNumPartitions())
print("\nThere are %d partitions in the voter_df_single DataFrame.\n" % voter_df_single.rdd.getNumPartitions())

# Add a ROW_ID field to each DataFrame
voter_df = voter_df.withColumn('ROW_ID', F.monotonically_increasing_id())
voter_df_single = voter_df_single.withColumn('ROW_ID', F.monotonically_increasing_id())

# Show the top 10 IDs in each DataFrame 
voter_df.orderBy(voter_df.ROW_ID.desc()).show(10)
voter_df_single.orderBy(voter_df_single.ROW_ID.desc()).show(10)

# Determine the highest ROW_ID and save it in previous_max_ID
previous_max_ID = voter_df_march.select('ROW_ID').rdd.max()[0]

# Add a ROW_ID column to voter_df_april starting at the desired value
voter_df_april = voter_df_april.withColumn('ROW_ID', previous_max_ID + F.monotonically_increasing_id())

# Show the ROW_ID from both DataFrames and compare
voter_df_march.select('ROW_ID').show()
voter_df_april.select('ROW_ID').show()

#Caching
    #stores DF in memory or on disk
    # improves speed on later transformations/actions
    #reduces resource usage
    #very large datasets may not fit in memory
    #local disk may not be a performance improvement
    #cached objects may not be available
    #cache only if you need it
    #try caching at different poitns and determine if performance improves
    #try to cache in memory or fast SSD/NVMe storage
    #can use intermediate parquet files
    #stop caching objects when finished
    #call .cache() on DF before action
    #caching is a transformation and not actually cached until an action
voter_df = spark.read.csv('voter_data.txt.gz')
voter_df.cache().count()
#Another way:
voter_df = voter_df.withColumn('ID', monotonically_increasing_id())
voter_df = voter_df.cache()
voter_df.show()

    #to check status .is_cached()
    #to get rid of cache - .unpersist() - on DF with no arguments

#Exercises:
start_time = time.time()

# Add caching to the unique rows in departures_df
departures_df = departures_df.distinct().cache()

# Count the unique rows in departures_df, noting how long the operation takes
print("Counting %d rows took %f seconds" % (departures_df.count(), time.time() - start_time))

# Count the rows again, noting the variance in time of a cached DataFrame
start_time = time.time()
print("Counting %d rows again took %f seconds" % (departures_df.count(), time.time() - start_time))

# Determine if departures_df is in the cache
print("Is departures_df cached?: %s" % departures_df.is_cached)
print("Removing departures_df from cache")

# Remove departures_df from the cache
departures_df.unpersist()

# Check the cache status again
print("Is departures_df cached?: %s" % departures_df.is_cached)

#Import performance
    #Spark cluster have 2 types of processes:
        #Driver Process (1)
        #Multiple worker processes
            #handle the actual transformations
    #Important things to consider:
        # Number of objects (files, network locations,etc.)
        # more objects are better than larger ones (easier to divide job)
        # can import via wildcard
airport_df = spark.read.csv('airports-*.txt.gz')
        # performs better if objects are around the same size
        #well defined schema will drastically improve import performance
            #avoids reading the data multiple times
            #provides validation on import
# Splitting objects
    #OS Utilities /scripts (split, cut, awk)
    #split -l 10000 -d largefile chunk-
    #can use custom scripts
    #can write it out to Parquet
df_csv = spark.read.csv('singlelargefile.csv')
df_csv.write.parquet('data.parquet')
df = spark.read.parquet('data.parquet')

#Exercises:
# Import the full and split files into DataFrames
full_df = spark.read.csv('departures_full.txt.gz')
split_df = spark.read.csv('departures_0*.txt.gz')

# Print the count and run time for each DataFrame
start_time_a = time.time()
print("Total rows in full DataFrame:\t%d" % full_df.count())
print("Time to run: %f" % (time.time() - start_time_a))

start_time_b = time.time()
print("Total rows in split DataFrame:\t%d" % split_df.count())
print("Time to run: %f" % (time.time() - start_time_b))

#Cluster sizing tips
# to read configuration settings:
spark.conf.get('configuration name')
# to write configs:
spark.conf.set('configuration name')
# spark deployment options:
    #single node
    #standalone
    #managed (by YARN, Mesos, K8S)
#Driver (one driver per cluster)
    #handles task assignment
    #result consolidation
    #shared data access
    #Tips:
        #driver node should have double the memory of the worker nodes
        #fast local storage is helpful
# Worker
    #runs actual tasks
    #ideally has all code, data, and resources for their given task
    # Recommendations:  
        #more worker nodes is often better than larger workers
        #especially important for import and export
        #test different configurations to find the best balance
        #fast local storage extremely useful
#Exercises:
# Name of the Spark application instance
app_name = spark.conf.get('spark.app.name')

# Driver TCP port
driver_tcp_port = spark.conf.get('spark.driver.port')

# Number of join partitions
num_partitions = spark.conf.get('spark.sql.shuffle.partitions')

# Show the results
print("Name: %s" % app_name)
print("Driver TCP port: %s" % driver_tcp_port)
print("Number of partitions: %s" % num_partitions)

# Store the number of partitions in variable
before = departures_df.rdd.getNumPartitions()

# Configure Spark to use 500 partitions
spark.conf.set('spark.sql.shuffle.partitions', 500)

# Recreate the DataFrame using the departures data file
departures_df = spark.read.csv('departures.txt.gz').distinct()

# Print the number of partitions for each instance
print("Partition count before change: %d" % before)
print("Partition count after change: %d" % departures_df.rdd.getNumPartitions())

#Spark Execution Plan
voter_df = df.select(df['VOTER NAME']).distinct()
voter_df.explain()
#result is the estimated plan to give the result from the data frame
#Shuffling - moving data around to various workers to complete a task
    #hides complexity from the user
    #can be slow to complete
    #lowers overall throughput
    #is often necessary, but try to minimize
#How to limit shuffling?
    #limit use of .repartition(num_partitions)
    #use coalesce(num_partitions) instead
    #use care when calling .join()
    #use broadcast()      
    # depending on use case, may not need to limit it at all  
#Broadcasting - provides a copy of an object to each worker
    #if each worker has its own copy, there is less need for communication between nodes
    #limits data shuffles
    #will be able to finish some tasks independently
    #can drastically speed up .join() operations
from pyspark.sql.functions import broadcast
combined_df = df_1.join(broadcast(df_2))

#Exercises:
# Join the flights_df and aiports_df DataFrames
normal_df = flights_df.join(airports_df, \
    flights_df["Destination Airport"] == airports_df["IATA"] )

# Show the query plan
normal_df.explain()

# Import the broadcast method from pyspark.sql.functions
from pyspark.sql.functions import broadcast

# Join the flights_df and airports_df DataFrames using broadcasting
broadcast_df = flights_df.join(broadcast(airports_df), \
    flights_df["Destination Airport"] == airports_df["IATA"] )

# Show the query plan and compare against the original
broadcast_df.explain()

start_time = time.time()
# Count the number of rows in the normal DataFrame
normal_count = normal_df.count()
normal_duration = time.time() - start_time

start_time = time.time()
# Count the number of rows in the broadcast DataFrame
broadcast_count = broadcast_df.count()
broadcast_duration = time.time() - start_time

# Print the counts and the duration of the tests
print("Normal count:\t\t%d\tduration: %f" % (normal_count, normal_duration))
print("Broadcast count:\t%d\tduration: %f" % (broadcast_count, broadcast_duration))

#Data Pipelines
    #set of steps to process data from source(s) to final output
    #can span many systems
    #inputs > transformations > (validation, analysis) > outputs
    #input - CSV, JSON, web services, db's
    #transformations - add columns, filter rows, drop nulls, calculations
    #output - CSV, Parquet, db
    #validation - checks that data is as we would like it
    #Analysis - calculations user would like (counts, etc.)
schema = StructType([StructField('name', StringType(), False), StructField('age', StringType(), False)
])  
df = spark.read.format('csv').load('datafile').schema(schema)
df = df.withColumn('id', monotonoically_increasing_id())
...
df.write.parquuet('outdata.parquet')
df.write.json('outdata.json')

# Import the data to a DataFrame
departures_df = spark.read.csv('2015-departures.csv.gz', header=True)

# Remove any duration of 0
departures_df = departures_df.filter(departures_df[3] > 0)

# Add an ID column
departures_df = departures_df.withColumn('id', F.monotonically_increasing_id())

# Write the file out to JSON format
departures_df.write.json('output.json', mode='overwrite')

#How to parse unconventional data
    #common issues:
        #incorrect data
        #commented lines
        #headers
        #nested structures with different delimiters
        #non-regular data
#Stanford ImageNet annotations example (dog breeds)
# Spark's CSV parser
    # automatically removes blank lines
    # comments can be removed with an optional argument
df1 = spark.read.csv('datafile.csv', comment='#')
    #headers defined via argument
    #ignored if a schema is defined
df1 = spark.read.csv('datafile.csv', header=True)
#Automatic column creation
    # automatically creates columns in a DF based on sep argument
df1 = spark.read.csv('datafile.csv', sep=',')
    #defaults to ,
    #can still successfully parse if sep is not in string
df1 = spark.read.csv('datafile.csv', sep="*")
    #stores data in column defaulting to _c0
    #allows you to properly handle nested separators

#Exercises
# Import the file to a DataFrame and perform a row count
annotations_df = spark.read.csv('annotations.csv.gz', sep='|')
full_count = annotations_df.count()

# Count the number of rows beginning with '#'
comment_count = annotations_df.filter(col('_c0').startswith('#')).count()

# Import the file to a new DataFrame, without commented rows
no_comments_df = spark.read.csv('annotations.csv.gz', sep='|', comment='#')

# Count the new DataFrame and verify the difference is as expected
no_comments_count = no_comments_df.count()
print("Full count: %d\nComment count: %d\nRemaining count: %d" % (full_count, comment_count, no_comments_count))

# Split _c0 on the tab character and store the list in a variable
tmp_fields = F.split(annotations_df['_c0'], '\t')

# Create the colcount column on the DataFrame
annotations_df = annotations_df.withColumn('colcount', F.size(tmp_fields))

# Remove any rows containing fewer than 5 fields
annotations_df_filtered = annotations_df.filter(~ (annotations_df["colcount"] < 5))

# Count the number of rows
final_count = annotations_df_filtered.count()
print("Initial count: %d\nFinal count: %d" % (initial_count, final_count))

# Split the content of _c0 on the tab character (aka, '\t')
split_cols = F.split(annotations_df['_c0'], '\t')

# Add the columns folder, filename, width, and height
split_df = annotations_df.withColumn('folder', split_cols.getItem(0))
split_df = split_df.withColumn('filename', split_cols.getItem(1))
split_df = split_df.withColumn('width', split_cols.getItem(2))
split_df = split_df.withColumn('height', split_cols.getItem(3))

# Add split_cols as a column
split_df = split_df.withColumn('split_cols', split_cols)

def retriever(cols, colcount):
  # Return a list of dog data
  return cols[4:colcount]

# Define the method as a UDF
udfRetriever = F.udf(retriever, ArrayType(StringType()))

# Create a new column using your UDF
split_df = split_df.withColumn('dog_list', udfRetriever(split_df.split_cols, split_df.colcount))

# Remove the original column, split_cols, and the colcount
split_df = split_df.drop('_c0').drop('split_cols').drop('colcount')

#Data Validation
    #verifying dataset complies with expected format
    #num of rows/columns
    #data types (compared against schema)
    #regex/complex rules
#validating via joins
    #compares data against known values
    #easy to find data in a given set
    #comparatively fast
parsed_df = spark.read.parquet('parsed_data.parquet')
company_df = spark.read.parquet('companies.parquet')
verified_df = parsed_df.join(company_df, parsed_df.company == company_df.company)
    #this automaticcally removes any rows with a company not in the valid_df
#Complex rule validation
    # calculations
    # verifying against an external source
    # likely uses a UDF to modify/verify

#Exercises:
# Rename the column in valid_folders_df
valid_folders_df = valid_folders_df.withColumnRenamed('_c0', 'folder')

# Count the number of rows in split_df
split_count = split_df.count()

# Join the DataFrames
joined_df = split_df.join(F.broadcast(valid_folders_df), "folder")

# Compare the number of rows remaining
joined_count = joined_df.count()
print("Before: %d\nAfter: %d" % (split_count, joined_count))

# Determine the row counts for each DataFrame
split_count = split_df.count()
joined_count = joined_df.count()

# Create a DataFrame containing the invalid rows
invalid_df = split_df.join(F.broadcast(joined_df), 'folder', 'left_anti')

# Validate the count of the new DataFrame is as expected
invalid_count = invalid_df.count()
print(" split_df:\t%d\n joined_df:\t%d\n invalid_df: \t%d" % (split_count, joined_count, invalid_count))

# Determine the number of distinct folder rows removed
invalid_folder_count = invalid_df.select('folder').distinct().count()
print("%d distinct invalid folders found" % invalid_folder_count)

#Analysis Calculations (UDF)
udfGetAvgSale = udf(getAvgSale, DoubleType())
df = df.withColumn('avg_sale', udfGetAvgSale(df.sales_list))
#UDF's have a performance hit compared to build-in Spark calculations
#try to perform calculations inline if possible
df = df.read.csv('datafile')
df = df.withColumn('avg', (df.total_sales / df.sales_count))
df = df.withColumn('sq_ft', df.width * df.length)
df = df.withColumn('total_avg_seze', udfComputeTotal(df.entries) / df.numEntries)

#Exercises:
# Select the dog details and show 10 untruncated rows
print(joined_df.select('dog_list').show(10, truncate=False))

# Define a schema type for the details in the dog list
DogType = StructType([
	StructField("breed", StringType(), False),
    StructField("start_x", IntegerType(), False),
    StructField("start_y", IntegerType(), False),
    StructField("end_x", IntegerType(), False),
    StructField("end_y", IntegerType(), False)
])

# Create a function to return the number and type of dogs as a tuple
def dogParse(doglist):
  dogs = []
  for dog in doglist:
    (breed, start_x, start_y, end_x, end_y) = dog.split(',')
    dogs.append((breed, int(start_x), int(start_y), int(end_x), int(end_y)))
  return dogs

# Create a UDF
udfDogParse = F.udf(dogParse, ArrayType(DogType))

# Use the UDF to list of dogs and drop the old column
joined_df = joined_df.withColumn('dogs', udfDogParse('dog_list')).drop('dog_list')

# Show the number of dogs in the first 10 rows
joined_df.select(F.size('dogs')).show(10)

# Define a UDF to determine the number of pixels per image
def dogPixelCount(doglist):
  totalpixels = 0
  for dog in doglist:
    totalpixels += (dog[3] - dog[1]) * (dog[4] - dog[2])
  return totalpixels

# Define a UDF for the pixel count
udfDogPixelCount = F.udf(dogPixelCount, IntegerType())
joined_df = joined_df.withColumn('dog_pixels', udfDogPixelCount('dogs'))

# Create a column representing the percentage of pixels
joined_df = joined_df.withColumn('dog_percent', (joined_df.dog_pixels / (joined_df.width * joined_df.height)) * 100)

# Show the first 10 annotations with more than 60% dog
joined_df.where('dog_percent > 60').show(10)