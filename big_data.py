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

#PYSPARK dataframes
    #immutable distribution of data with named columns
    # can rpocess structured (relational database) or semi structured (JSON)
    # supports Python, R, Scala, Java
    # can you SQL statements (SELECT * from table) or expression methods (df.select())
    # Spark Session is point of entry to interact with Spark DF
    # Spark session can be used to create df, register df, execute SQL queries
    # available in PySpark shell as spark
    # Two ways to create df's
        # from existing RDD's using SparkSession createDateFrame() method
        # from various data sources (CSV, JSON, TXT) using read method
    # Schemas - control the data and helps DF to optimize queries
        #provide column names, type of data, whether null values allowed, etc.

iphones_RDD = sc.parallelize([
    ("XS", 2018, 5.65, 2.79, 6.24),
    ("XR", 2018, 5.94, 2.98, 6.84)
])
names = ["Model", "Year", "Height", "Width", "Weight"]
iphones_df = spark.createDataFrame(iphones_RDD, schema=names)

#creating a df from a file - with two optional parameters
df_csv = spark.read.csv("people.csv", header=True, inferSchema=True)
df_json = spark.read.json("people.json", header=True, inferSchema=True)

#Exercises:
# Create an RDD from the list
rdd = sc.parallelize(sample_list)

# Create a PySpark DataFrame
names_df = spark.createDataFrame(rdd, schema=['Name', 'Age'])

# Check the type of names_df
print("The type of names_df is", type(names_df))

# Create an DataFrame from file_path
people_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the type of people_df
print("The type of people_df is", type(people_df))

#DF Transformations:
    #select(), filter(), groupby(), orderby(), dropDuplicates(), withColumnRenamed()
    #printSchema() - method for any Spark dataset
#DF Actions:
    # head(), show(), count(), columns, describe()

    #select() - subsets the columns in the DF
    #show() - prints first 20 rows by default
df_id_age = test.select('Age')
df_id_age.show(3)
    #filter() - filters out the rows based on a condition
new_df_age21 = new_df.filter(new_df.Age > 21)
new_df_age21.show(3)
    #groupby() - used to group a variable by column (this is kind of like a groupby dictionary that counts)
test_df_age_group = test_df.groupby('Age')
test_df_age_group.count().show(3)
    #orderby() - returns a df sorted based on one or more columns
test_df_age_group.count().orderBy('Age').show(3)
    #dropDuplicates - returns a new df with duplicates removed
test_df_no_dup = test_df.select('User_ID', 'Gender' 'Age').dropDuplicates()
test_df_no_dup.count()
    #withColumnRenamed() - renames a column in DF
test_df_sex = test_df.withColumnRenamed('Gender', 'Sex')
test_df_sex.show(3)
    #printSchema() - printe the types of columns in df
    #columns - printe the names of the columns
test_df.columns
    #describe() - prints summary statistics of numerical columns
test_df.describe().show()

#Exercises:
# Print the first 10 observations 
people_df.show(10)

# Count the number of rows 
print("There are {} rows in the people_df DataFrame.".format(people_df.count()))

# Count the number of columns and their names
print("There are {} columns in the people_df DataFrame and their names are {}".format(len(people_df.columns), people_df.columns))

# Select name, sex and date of birth columns
people_df_sub = people_df.select('name', 'sex', 'date of birth')

# Print the first 10 observations from people_df_sub
people_df_sub.show(10)

# Remove duplicate entries from people_df_sub
people_df_sub_nodup = people_df_sub.dropDuplicates()

# Count the number of rows
print("There were {} rows before removing duplicates, and {} rows after removing duplicates".format(people_df_sub.count(), people_df_sub_nodup.count()))

# Filter people_df to select females 
people_df_female = people_df.filter(people_df.sex == "female")

# Filter people_df to select males
people_df_male = people_df.filter(people_df.sex == "male")

# Count the number of rows 
print("There are {} rows in the people_df_female DataFrame and {} rows in the people_df_male DataFrame".format(people_df_female.count(), people_df_male.count()))

#Why use DF API vs. SQL queries
    #easier to construct programmatically
    #SQL queries can be concise and easier to understand and are portable
    #Spark Session has sql() that executes SQL queries
    #takes sql statement as an argument and returns the result as a DF
    #cannot be run directly against a DF
df.createsOrReplaceTempView("table1")   #creates a templated table that you can run sql against
df2 = spark.sql("SELECT field1, field2 FROM table1")
df2.collect()

#Exercises:
# Create a temporary table "people"
people_df.createOrReplaceTempView("people")

# Construct a query to select the names of the people from the temporary table "people"
query = '''SELECT name FROM people'''

# Assign the result of Spark's query to people_df_names
people_df_names = spark.sql(query)

# Print the top 10 names of the people
people_df_names.show(10)

# Filter the people table to select female sex 
people_female_df = spark.sql('SELECT * FROM people WHERE sex=="female"')

# Filter the people table DataFrame to select male sex
people_male_df = spark.sql('SELECT * FROM people WHERE sex=="male"')

# Count the number of rows in both DataFrames
print("There are {} rows in the people_female_df and {} rows in the people_male_df DataFrames".format(people_female_df.count(), people_male_df.count()))

#Data Visualization in PySpark
    #can't use matplotlib, Seaborn or Bokeh
    #Can use:
        #pyspark_dist_explore library - quick insights (hist(), distplot(), pandas_histogram())
test_df = spark.read.csv("test.csv", header=True, inferSchema=True)
test_df_age = test_df.select('Age')
hist(test_df_age, bins=20, color='red')
        #toPandas() - converts to Pandas df, then you can use matplotlib
test_df = spark.read.csv("test.csv", header=True, inferSchema=True)
test_df_sample_pandas = test_df.toPandas()
test_df_sample_pandas.hist('Age')
        #HandySpark library - meant to help with eda
test_df = spark.read.csv("test.csv", header=True, inferSchema=True)
hdf = test_df.toHandy()
hdf.cols["Age"].hist()

#Pandas vs. PySpark
    #Pandas are in-memory, single server (size limited by server memory)
    #pySpark are run in parallel, lazy evaluation
    #pandas are mutable, pyspark are immutable
    #pandas offers more operations than pyspark

#Exercises:
# Check the column names of names_df
print("The column names of names_df are", names_df.columns)

# Convert to Pandas DataFrame  
df_pandas = names_df.toPandas()

# Create a horizontal bar plot
df_pandas.plot(kind='barh', x='Name', y='Age', colormap='winter_r')
plt.show()

# Load the Dataframe
fifa_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Check the schema of columns
fifa_df.printSchema()

# Show the first 10 observations
fifa_df.show(10)

# Print the total number of rows
print("There are {} rows in the fifa_df DataFrame".format(fifa_df.count()))

# Create a temporary view of fifa_df
fifa_df.createOrReplaceTempView('fifa_df_table')

# Construct the "query"
query = '''SELECT "Age" FROM fifa_df_table WHERE Nationality == "Germany"'''

# Apply the SQL "query"
fifa_df_germany_age = spark.sql(query)

# Generate basic statistics
fifa_df_germany_age.describe().show()

# Convert fifa_df to fifa_df_germany_age_pandas DataFrame
fifa_df_germany_age_pandas = fifa_df_germany_age.toPandas()

# Plot the 'Age' density of Germany Players
fifa_df_germany_age_pandas.plot(kind='density')
plt.show()

#PySpark MLlib
    # Tools: ML Algorithms (collaborative filtering, classification, clustering) 
    # Featurization: feature extraction, transformation, dimensionality reduction, selection
    # Pipelines: tools for constructing, evaluating, and tuning ML pipelines
    # sci-kit learn is great for small datasets on a single machine
    # MLlib algorithms are designed for parallel processing on a cluster
    # Supports Scala, Java, R
    # Good for iterative algorithms
    # Classification (binary and multiclass) and regression): Linear SVM's, logistic regression,
        #decision trees, random forests, gradient boosted trees, naive Bayes, linear least squares, 
        #Lasso, ridge regression, isotonic regression
    # Collaborative filtering: alternating least squares (ALS)
    # Clustering - K-means, Gaussian mixture, Bisecting K-means, Streaming K-means
    # Three C's:
        # Collaborative filtering (recommender engines): produce recommendations 
from pyspark.mllib.recommendation import ALS
        # Classification - identifying categories for a new observation
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
        # CLustering - groups data based on similar characteristics
from pyspark.mllib.clustering import KMeans

#Collaborative filtering - finding users that share common interests
    #User/user approach - finds users similar to target user
    #Item/item approach - finds items a that are similar to items associated with the target user
    #Rating class - wrapper around tuple(user, product, rating)
    #Useful for parsing RDD and creating a tuple of user, product, rating
from pyspark.mllib.recommendation import Rating
r = Rating(user = 1, product = 2, rating = 5.0)
#Splitting the data using randomSplit()
    #splits for test/train and returns multiple RDD's
data = sc.parallelize([1,2,...])
training, test = data.randomSplit([0.6, 0.4])
training.collect()
test.collect() 
    #ALS.train(ratings, rank, iterations)
ratings = sc.parallelize([r1, r2, r3])
model = ALS.train(ratings, rank=10, iterations=10)
    #predictAll() - returns RDD of rating objects (predicts ratings for input user and product pair)
unrated_RDD = sc.parallelize([(1,2), (1,1)])
predictions = model.predictAll(unrated_RDD)
predictions.collect()
    #Model evaluation with Mean Square Error (MSE)
        #average value of the equare of (actual rating - predicted rating)
rates = ratings.map(lambda x: ((x[0], x[1]), x[2]))
rates.collect()
preds = predictions.map(lambda x: ((x[0], x[1]), x[2]))    
preds.collect()
rates_preds = rates.join(preds)
rates_preds.collect()
MSE = rates_preds.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean()

#Exercises:
# Load the data into RDD
data = sc.textFile(file_path)

# Split the RDD 
ratings = data.map(lambda l: l.split(','))

# Transform the ratings RDD 
ratings_final = ratings.map(lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))

# Split the data into training and test
training_data, test_data = ratings_final.randomSplit([0.8, 0.2])

# Create the ALS model on the training data
model = ALS.train(training_data, rank=10, iterations=10)

# Drop the ratings column 
testdata_no_rating = test_data.map(lambda p: (p[0], p[1]))

# Predict the model  
predictions = model.predictAll(testdata_no_rating)

# Return the first 2 rows of the RDD
predictions.take(2)

# Prepare ratings data
rates = ratings_final.map(lambda r: ((r[0], r[1]), r[2]))

# Prepare predictions data
preds = predictions.map(lambda r: ((r[0], r[1]), r[2]))

# Join the ratings data with predictions data
rates_and_preds = rates.join(preds)

# Calculate and print MSE
MSE = rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error of the model for the test data = {:.2f}".format(MSE))

#Classification
# predicts categories (like spam filter)
# uses supervised learning to sort the input data into different categories
# binary (spam or not)
# multi-class (more than 2 categories i.e. categorizing new articles)
#Logistic Regression - predicts a binary response based on some variables
    #measures relationship from label on y axis and features on the x axis
    # probability 0-1 (usually .5 ad up is predicted to 1)
# Vectors
    # 2 types: 
        #Dense: store all their entries in an array of floating point numbers
        #Sparse: stores only nonzero values and their indices

denseVec = Vectors.dense([1.0, 2.0, 3.0])
sparseVec = Vectors.sparse(4, {1:1.0, 3:5.5})

#LabeledPoint - wrapper for input features and predicted value
    #for binary classification of log reg, a label is either 0 or 1
positive = LabeledPoint(1.0, [1.0, 0.0, 3.0])
negative = LabeledPoint(0.0, [2.0, 1.0, 3.0])

#HashingTF - used to map feature value to indices in feature vector
from pyspark.mllib.feature import HashingTF
sentence = "hello hello world"
words = sentence.split()
tf = HashingTF(10000)
tf.transform(words)

data = [
    LabeledPoint(1.0, [1.0, 0.0, 3.0]),
    LabeledPoint(0.0, [2.0, 1.0, 3.0])
]
RDD = sc.parallelize(data)
lrm = LogisticRegressionWithLBFGS.train(RDD)
lrm.predict([1.0, 0.0])
lrm.predict([0.0, 1.0])

#Exercises:
# Load the datasets into RDDs
spam_rdd = sc.textFile(file_path_spam)
non_spam_rdd = sc.textFile(file_path_non_spam)

# Split the email messages into words
spam_words = spam_rdd.flatMap(lambda email: email.split(' '))
non_spam_words = non_spam_rdd.flatMap(lambda email: email.split(' '))

# Print the first element in the split RDD
print("The first element in spam_words is", spam_words.first())
print("The first element in non_spam_words is", non_spam_words.first())

# Create a HashingTf instance with 200 features
tf = HashingTF(numFeatures=200)

# Map each word to one feature
spam_features = tf.transform(spam_words)
non_spam_features = tf.transform(non_spam_words)

# Label the features: 1 for spam, 0 for non-spam
spam_samples = spam_features.map(lambda features:LabeledPoint(1, features))
non_spam_samples = non_spam_features.map(lambda features:LabeledPoint(0, features))

# Combine the two datasets
samples = spam_samples.join(non_spam_samples)

# Split the data into training and testing
train_samples,test_samples = samples.randomSplit([0.8, 0.2])

# Train the model
model = LogisticRegressionWithLBFGS.train(train_samples)

# Create a prediction label from the test data
predictions = model.predict(test_samples.map(lambda x: x.features))

# Combine original labels with the predicted labels
labels_and_preds = test_samples.map(lambda x: x.label).zip(predictions)

# Check the accuracy of the model on the test data
accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_samples.count())
print("Model accuracy : {:.2f}".format(accuracy))

# Clustering
# unsupervised learning to cluster unlabeled data
# supports K-means (bisecting, streaming too), Gaussian mixture, power iteration clustering, 
# K means in the most popular
# give data points and a certain # of clusters
# need to be numerical feaatures
RDD - sc.textFile("WineData.csv").map(lambda x: x.split(",")).map(lambda x: [float(x[0]), float(x[1])])
RDD.take(5)

from pyspark.mllib.clustering import KMeans
model = KMeans.train(RDD, k=2, maxIterations = 10)
model.clusterCenters

# does not have a built in evaluation method
from math import sqrt
def error(point):
    center = model.centers[model.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSE = RDD.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = ") + str(WSSSE)

#visualizing KMeans
import matplotlib as plt
wine_data_df = spark.createDataFrame(RDD, schema=["col1", "col2"])
wine_data_df_pandas = wine_data_df.toPandas()
cluster_centers_pandas = pd.DataFrame(model.clusterCenters, columns=["col1", "col2"])
cluster_centers_pandas.head()
plt.scatter(wine_data_df_pandas["col1"], wine_data_df_pandas["col2"])
plt.scatter(cluster_centers_pandas["col1"], cluster_centers_pandas["col2"], color="red", marker="x")

# Exercises:
# Load the dataset into an RDD
clusterRDD = sc.textFile(file_path)

# Split the RDD based on tab
rdd_split = clusterRDD.map(lambda x: x.split("\t"))

# Transform the split RDD by creating a list of integers
rdd_split_int = rdd_split.map(lambda x: [int(x[0]), int(x[1])])

# Count the number of rows in RDD 
print("There are {} rows in the rdd_split_int dataset".format(rdd_split_int.count()))

# Train the model with clusters from 13 to 16 and compute WSSSE
for clst in range(13, 17):
    model = KMeans.train(rdd_split_int, clst, seed=1)
    WSSSE = rdd_split_int.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("The cluster {} has Within Set Sum of Squared Error {}".format(clst, WSSSE))

# Train the model again with the best k
model = KMeans.train(rdd_split_int, k=15, seed=1)

# Get cluster centers
cluster_centers = model.clusterCenters

# Convert rdd_split_int RDD into Spark DataFrame and then to Pandas DataFrame
rdd_split_int_df_pandas = spark.createDataFrame(rdd_split_int, schema=["col1", "col2"]).toPandas()

# Convert cluster_centers to a pandas DataFrame
cluster_centers_pandas = pd.DataFrame(cluster_centers, columns=["col1", "col2"])

# Create an overlaid scatter plot of clusters and centroids
plt.scatter(rdd_split_int_df_pandas["col1"], rdd_split_int_df_pandas["col2"])
plt.scatter(cluster_centers_pandas["col1"], cluster_centers_pandas["col2"], color="red", marker="x")
plt.show()

