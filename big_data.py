# Three V's of Big Data
    # Volume - size of the data
    # Variety - different sources and formats
    # Velocity - speed that data is generated and available for processing

# Clustered computing - collection of resources of multiple machines
# Parallel computing - type of computation carried out simultaneously
# Distributed computing - nodes or networked computers that run jobs in parallel
# Batch processing - breaking data into smaller pieces and running each piece on an
    # individual machine
# Real time processing - demands that information is processed and made ready 
    # immediately

# Processing Systems
    # Hadoop/MapReduce - scalable fault tolerant written in Java
        # open source, good for batch processing
    # Apache Spark - general purpose lighnting fast cluster computing system
        #framework for storing and processing across clustered computers
        # open source, good for batch processing and real time data processing

# Apache Spark
    # distributes data and computation across a distributed cluster
    # executes complex multi-stage applications such as ML
    # efficient in-memory computations for large data sets
    # Very fast
    # Spark is written in Scala, but supports Java, Python, R and SQL
    # Spark Core - contains the basic functionality of Spark with libraries built on top
#Spark Libraries
    # Spark SQL - processes structured and semi-structured data in Python, Java, and Scala
    # MLib - library of common machine learning algorithms
    # GraphX - collection of algorithms for manipulating graphs and performing parallel graph computations
    # Spark Streaming - scalable, high throughput processing library for real time data
# Spark Modes
    # Local mode - single machine (like your laptop)
        # convenient for testing, debugging, and demonstration
    # Cluster mode - set of predefined machines
        # mainly used for production
    # Typical workflow is that you start in local and transition to cluster - no code change necessary

#PySpark
    # similar computational speed and power as Scala
    # API's similar to Pandas and scikit learn packages
#Spark Shell
    # comes with interactive shells that enable ad-hoc data analysis
    # interactive environment through which one can access Spark's functionality quickly and conveniently
    # helpful for fast interactive prototyping before running the jobs on clusters
    # allows you to interact iwth data on disk or in memoty across many machines
    # takes care of automatically distributing this processing
    # available in Spark-shell (Scala), PySpark (Python), SparkR (R)
    # PySpark shell - python based command line tool
    # allows interface with spark data structures
    # supports connecting to a cluster
# Spark Context
    # entry point to interact with underlying Spark functionality
    # entry point - where control is transfered from OS to the provided program
    # accessed in PySpark shell with sc
    # sc.version - displays the version currently running
    # sc.pythonVer - prints version of Python currently being used
    # sc. master - URL of the cluster or local string to run in local mode
# Loading Data
    # creates parallel collections
        # rdd = sc.parallelize([1,2,3,4])
    # textFile() method
        # rdd2 = sc.textFile("text.txt")
# Exercises:
# Create a Python list of numbers from 1 to 100 
numb = range(1, 100)

# Load the list into PySpark  
spark_data = sc.parallelize(numb)

# Load a local file into PySpark shell
lines = sc.textFile(file_path)

# Functional programming in Python
# anonymous functions - functions not bound to a name at runtime (lambda functions)
    # often used with map() and filter()
    # creates a function to be called later in the program
    # returns function instead of assigning it to a name
    # way to inline a function definition or defer execution of code
    # can be used as function objects
    # can have any number of arguments, but only one expression (which is evaluated and returned)
    # lambda arguments: expression
double = lambda x: x * 2
print(double(3))

def cube(x):
    return x ** 3
g = lambda x: x ** 3
print(g(10))
print(cube(10))
    #lambda definition does not include a return statement, it always contains an expression which is returned
    # can use lambda anywhere a function is expected, and don't have to assign it to a variable at all
# map() functions - takes a function and a list and returns a new list which contains the items returned by 
    # that functions for each item
    # map(function, list)
items = [1,2,3,4]
list(map(lambda x: x + 2, items))
# filter() function - takes a function and a list and returns a new list for which the function evaluates as true
    # filter(function, list)
items = [1,2,3,4]
list(filter(lambda x: (x%2 != 0), items))

#Exercises:
# Print my_list in the console
print("Input list is", my_list)

# Square all numbers in my_list
squared_list_lambda = list(map(lambda x: x ** 2, my_list))

# Print the result of the map function
print("The squared numbers are", squared_list_lambda)

# Print my_list2 in the console
print("Input list is:", my_list2)

# Filter numbers divisible by 10
filtered_list = list(filter(lambda x: (x%10 == 0), my_list2))

# Print the numbers divisible by 10
print("Numbers divisible by 10 are:", filtered_list)

#Abstracting data with PySpark RDD
#RDD - resilient distributed datasets (collection of data distributed across cluster)
# fundamental and bacbone data type in PySpark
# Spark driver creates RDD and distributes the data across cluster nodes (with each node
#   containing a slice of data)
# Resilient - ability to withstand failures and recompute missing or damaged partitions
# Distributed - jobs span across multiple nodes for efficient computation
# Datasets - collection of partitions data (arrays, tables, typles, etc)
# Can make an RDD:
    # parallelizing existing collection of objects
    # external datasets (HDFS, S3 buckets, lines in text file)
    # from existing RDD's
# Partition - logical division of a large distributed data set, each part stored in multiple
    # locations across the cluster
    # partitioned - can define minimum # of partitions with 2nd argument (miniPartitions)
numRDD = sc.parallelize(range(10), minPartitions = 6)
fileRDD = sc.textFile("README.md", minPartitions = 6)

# Exercises:
# Create an RDD from a list of words
RDD = sc.parallelize(["Spark", "is", "a", "framework", "for", "Big Data processing"])

# Print out the type of the created object
print("The type of RDD is", type(RDD))

# Print the file_path
print("The file_path is", file_path)

# Create a fileRDD from file_path
fileRDD = sc.textFile(file_path)

# Check the type of fileRDD
print("The file type of fileRDD is", type(fileRDD))

# Check the number of partitions in fileRDD
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())

# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions = 5)

# Check the number of partitions in fileRDD_part
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())

# PySpark Operations
    # Transformations - operations on RDD's that return a new RDD
        # map(), filter(), flatMap(), union() 
    # Actions - operations that perform some computation on the RDD
        # collect(), take(N), reduce(), count()
    # Lazy evaluation- helps with fault tolerance and optimizing resources
RDD = sc.parallelize([1,2,3,4])
RDD_map = RDD.map(lambda x: x * x)

RDD_filter = RDD.filter(lambda x: x > 2)

# flapMap() - returns multiple values for each element in the original RDD
    # use case would be splitting strings into words
RDD_string = sc.parallelize(["Hello World", "How are you"])
RDD_flatMap = RDD_flatMap.flatMap(lambda x: x.split(" "))
# returns ["Hello", "world", "How", "are", "you"]

#Union combines two RDD's into one
inputRDD = sc.textFile("logs.txt")
errorRDD = inputRDD.filter(lambda x: "error" in x.split())
warningRDD = inputRDD.filter(lambda x: "warnings" in x.split())
combinedRDD = errorRDD.union(warningRDD)

#Actions return a value after running a computation on the RDD
# collect() - returns all elements of dataset as an array
# take(N) - returns an array with the first N elements of hte dataset
# first() - prints the first element of the RDD
# count() - returns the nubmer of elements in the RDD

# Exercises:
# Create map() transformation to cube numbers
cubedRDD = numbRDD.map(lambda x: x ** 3)

# Collect the results
numbers_all = cubedRDD.collect()

# Print the numbers from numbers_all
for numb in numbers_all:
	print(numb)

# Filter the fileRDD to select lines with Spark keyword
fileRDD_filter = fileRDD.filter(lambda line: 'Spark' in line)

# How many lines are there in fileRDD?
print("The total number of lines with the keyword Spark is", fileRDD_filter.count())

# Print the first four lines of fileRDD
for line in fileRDD_filter.take(4): 
  print(line)

# Pair RDD's 
# many real life datasets are key, value pairs
# each row is a key and maps to one or more values
# key is the identifier and value is the data
# Commonly created from a list of key value tuples or from a regular RDD
# need to get data into key/value form
my_tuple = [('sam', 23), ('mary', 34), ('peter', 25)]
pairRDD_tuple = sc.parallelize(my_tuple)

my_list = ['sam 23', 'mary 34', 'peter 25']
regularRDD = sc.parallelize(my_list)
pairRDD_RDD = regularRDD.map(lambda s: (s.split(' ')[0], s.split(' ')[1]))

# all regular transformations work on pair RDD
# have to pass functions that operate on key value pairs rather than individual elements
    # reduceByKey(func) transformation combine values with the same key
        #runs parallel operations for each key in the dataset
        # returns a new RDD with each key and the reduced value for that key
regularRDD = sc.parallelize([('Messi', 23), ("Ronaldo", 34), ("Neymar", 22), ("Messi, 24")])
pairRDD_reducebykey = regularRDD.reduceByKey(lambda x,y : x + y)
pairRDD_reducebykey.collect()
#returns [('Messi', 47), ("Ronaldo", 34), ("Neymar", 22)]
#combines like keys and this function then adds the values

# groupByKey() group values with the same key
airports = [("US", "JFK"), ("UK", "LHR"), ("FR", "CDG"), ("US", "SFO")]
regularRDD = sc.parallelize(airports)
pairRDD_group = regularRDD.groupByKey().collect()
for cont, air in pairRDD_group:
    print(cont, list(air))

# sortByKey() returns an RDD sorted by the key - can be ascending or descending
pairRDD_reducebykey_rev = pairRDD_reducebykey.map(lambda x: (x[1], x[0]))
pairRDD_reducebykey_rev.sortByKey(ascending=False).collect()

#join() joins the two pair RDDs based on their key
RDD1 = sc.parallelize([("Messi", 34), ("Ronaldo", 32), ("Neymar", 24)])
RDD2 = sc.parallelize([("Messi", 80), ("Ronaldo", 120), ("Neymar", 100)])
RDD1.join(RDD2).collect()
#returns [("Messi", (34, 80), ("Ronaldo", (32, 120), ("Neymar", (24, 100)]

#Exercises:
# Create PairRDD Rdd with key value pairs
Rdd = sc.parallelize([(1,2),(3,4),(3,6),(4,5)])

# Apply reduceByKey() operation on Rdd
Rdd_Reduced = Rdd.reduceByKey(lambda x, y: (x + y))

# Iterate over the result and print the output
for num in Rdd_Reduced.collect(): 
  print("Key {} has {} Counts".format(num[0], num[1]))

# Create PairRDD Rdd with key value pairs
Rdd = sc.parallelize([(1,2),(3,4),(3,6),(4,5)])

# Apply reduceByKey() operation on Rdd
Rdd_Reduced = Rdd.reduceByKey(lambda x, y: (x + y))

# Iterate over the result and print the output
for num in Rdd_Reduced.collect(): 
  print("Key {} has {} Counts".format(num[0], num[1]))

#RDD Actions
#reduce(func) - used for aggregating the elements of a regular RDD
    #func must be commutative and associative
x = [1,2,3]
RDD = sc.parallelize(x)
RDD.reduce(lambda x,y : x + y)      #this would sum and return it (6)

#saveAsTextFile() - when using collect on big data it can take too long
    # saves RDD into a text file inside directory with each partition as a separate file
RDD.saveAsTestFile("tempfile")
    #coalesce method can be used to return a single text file
RDD.coalesce(1).saveAsTextFile("tempfile")

#countByKey -only available for key (K,V) - count the # of elements for each key
rdd = sc.parallelize([("a", 1), ("b", 2), ("a", 1)])
for key, val in rdd.countByKey().items():
    print(key, val)
    #should only be used on datasets that can be stored in memory

#collectAsMap() - returns the key value pairs as a dictionary
sc.parallelize([(1,2), (3,4)]).collectAsMap()
    #returns {1:2, 3:4}
    #should only be used on datasets that can be stored in memory

#Exercises:
# Count the unique keys
total = Rdd.countByKey()

# What is the type of total?
print("The type of total is", type(total))

# Iterate over the total and print the output
for k, v in total.items(): 
  print("key", k, "has", v, "counts")

#Exercises:
# Create a baseRDD from the file path
baseRDD = sc.textFile(file_path)

# Split the lines of baseRDD into words
splitRDD = baseRDD.flatMap(lambda x: x.split())

# Count the total number of words
print("Total number of words in splitRDD:", splitRDD.count())

# Convert the words in lower case and remove stop words from the stop_words curated list
splitRDD_no_stop = splitRDD.filter(lambda x: x.lower() not in stop_words)

# Create a tuple of the word and 1 
splitRDD_no_stop_words = splitRDD_no_stop.map(lambda w: (w, 1))

# Count of the number of occurences of each word
resultRDD = splitRDD_no_stop_words.reduceByKey(lambda x, y: x + y)

# Convert the words in lower case and remove stop words from the stop_words curated list
splitRDD_no_stop = splitRDD.filter(lambda x: x.lower() not in stop_words)

# Create a tuple of the word and 1 
splitRDD_no_stop_words = splitRDD_no_stop.map(lambda w: (w, 1))

# Count of the number of occurences of each word
resultRDD = splitRDD_no_stop_words.reduceByKey(lambda x, y: x + y)