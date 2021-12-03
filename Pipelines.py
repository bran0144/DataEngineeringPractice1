# Components of a data platform
# Data Lake - typically comprises several systems and is typically organized in several zones
# data from operational systems ends up in a "landing zone" (raw data)
# ingestion - getting data into the data lake
# clean zone - data from landing zone and cleaned (to prevent many similar transformations of data)
# business zone - applies ml algorithms to cleaned data (domain specific data)
# to move data from one zone to another and transform it, data pipelines are built
# pipelines can be triggered by external events like files be stored, or scheduled, or even manually
# ETL - extract, transform, load pipelines
# To navigate a data lake, a data catalog is typically provided

# Data ingestion with Singer
# writes scripts that move data
# Singer is a specification that uses JSON as the data exchange format
# taps - extraction scripts
# targets - loads scripts
# language independent
# can be mixed and matched to create smaller data pipelines
# Taps and targets communicate over streams:
    # schema(metadata)
    # state(process metadata)
    # record(data)
# Stream - a named virtual location to which you send messages that can be picked up at a downstream location
# different streams cna be used to partition data based on the topic
# Singer spec - first you describe the data by specifying its schema
# schema should be valid JSON
json_schema = {
  "properties": {"age": {"maximum": 130,
                          "minimum": 1,
                          "type": "integer"},
              "has_children": {"type": "boolean"},
              "id": {"type": "integer"},
              "name": {"type": "string"},
  "$id": "http://yourdomain.com/schemas/my_user_schema.json",
  "$scheme": "http//json-schema.org/draft-08/sceham#"}}

# id and schema are optional, but high recommended
import singer
singer.write_schema(schema=json_schema, 
                    stream_name='DC-employees',
                    key_properties=["id"])

# if there is no primary key, specify an empty list
# write_schema - wraps actual JSON scehma into a new JSON message and adds a few attributes
# json.dumps - transforms object into a string
# json.dump - writes the string to a file

# Import json
import json

database_address = {
  "host": "10.0.0.5",
  "port": 8456
}

# Open the configuration file in writable mode
with open("database_config.json", "w") as fh:
  # Serialize the object in this file handle
  json.dump(obj=database_address, fp=fh)

# Complete the JSON schema
schema = {'properties': {
    'brand': {'type': 'string'},
    'model': {'type': 'string'},
    'price': {'type': 'number'},
    'currency': {'type': 'string'},
    'quantity': {'type': 'integer', 'minimum': 1},  
    'date': {'type': 'string', 'format': 'date'},
    'countrycode': {'type': 'string', 'pattern': "^[A-Z]{2}$"}, 
    'store_name': {'type': 'string'}}}

# Write the schema
columns = ("id", "name", "age", "has_children")
users = {(1, "John", 20, False),
        (2, "Mary", 35, True),
        (3, "Sophia", 25, False)}
singer.write_schema(stream_name="products", schema=schema, key_properties=[])
singer.write_record(stream_name="DC-employees", 
               record=dict(zip(columns, users.pop())))


# Running an ingestion pipeline
# to convert a user into a Singer RECORD message, we use the "write_record" function

singer.write_record(stream_name="DC_employees",
            record=dict(zip(columns, users.pop())))
# stream_name needs to match the stream you specificed in a schema message
# Singer does a few more transformations that make better JSON than using **

# can use the unpacking operator (**) - which unpacks a dictionary into another one
fixed_dict={"type": "RECORD", "stream": "DC_employees"}
record_msg={**fixed_dict, "record": dict(zip(columns, users.pop()))}
print(json.dumps(record_msg))

# If you have a Singer target, that can parse the messages (along with write_schema
# and write_record), then you have a full ingestion pipeline (using | )
# write_records - can take more than one record

# python my_tap.py | target-csv
# this creates csv files from the json lines
# will put csv in the same directory from where you run the command
# python my_tap.py | target-csv --config userconfig.cfg
# usually they are package in python so your call would look like this:
# my-packaged-tap | target-csv --config userconfig.cfg

# Allows for modualr ingestion pipelines
# my-packaged-tap | target-google-sheets
# my-packaged-tap | target-postgresql --config conf.json
# just need to change taps and targets and you can ingest easily

# State messages
# good for when you only want to read newest values (you can emit state at the end of a 
#   successful run)
# then reuse that same message to only get the new messages
singer.write_state(value={"max-last-updated-on": some_variable})

# Exercises:
endpoint = "http://localhost:5000"

# Fill in the correct API key
api_key = "scientist007"

# Create the web API’s URL
authenticated_endpoint = "{}/{}".format(endpoint, api_key)

# Get the web API’s reply to the endpoint
api_response = requests.get(authenticated_endpoint).json()
pprint.pprint(api_response)

# Create the API’s endpoint for the shops
shops_endpoint = "{}/{}/{}/{}".format(endpoint, api_key, "diaper/api/v1.0", "shops")
shops = requests.get(shops_endpoint).json()
print(shops)

# Create the API’s endpoint for items of the shop starting with a "D"
items_of_specific_shop_URL = "{}/{}/{}/{}/{}".format(endpoint, api_key, "diaper/api/v1.0", "items", "DM")
products_of_shop = requests.get(items_of_specific_shop_URL).json()
pprint.pprint(products_of_shop)

# Use the convenience function to query the API
tesco_items = retrieve_products("Tesco")

singer.write_schema(stream_name="products", schema=schema,
                    key_properties=[])

# Write a single record to the stream, that adheres to the schema
singer.write_record(stream_name="products", 
                    record={**tesco_items[0], "store_name": "Tesco"})

for shop in requests.get(SHOPS_URL).json()["shops"]:
    # Write all of the records that you retrieve from the API
    singer.write_records(
      stream_name="products", # Use the same stream name that you used in the schema
      records=({**item, "store_name": shop}
               for item in retrieve_products(shop))
    )

# Pyspark
# fast engine for large-scale data processing
# 4 libraries built on top of Spark core:
#   SparkSQL, Spark Streaming, MLib, GraphX
# Useful for data processing at scale
# Parallelizing its execution over multiple machines
# Useful for interactive analytics
# Not good for: when your data is small

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
prices = spark.read.options(header="true").csv("mnt/data_lae/landing/prices.csv")
prices.show()

# show displays the first 20 rows in tab format
from pprint import pprint
pprint(prices.dyptes)

# Spark can infer data types, but it is better to define the schema
schema = StructType([StructField("store", StringType(), nullable=False),
        StructField("countrycode", StringType(), nullable=False)])
prices = spark.read.options(header="true").schema(schema).csv("mnt/data_lae/landing/prices.csv")

# Define the schema
schema = StructType([
  StructField("brand", StringType(), nullable=False),
  StructField("model", StringType(), nullable=False),
  StructField("absorption_rate", ByteType(), nullable=True),
  StructField("comfort", ByteType(), nullable=True)
])

better_df = (spark
             .read
             .options(header="true")
             # Pass the predefined schema to the Reader
             .schema(schema)
             .csv("/home/repl/workspace/mnt/data_lake/landing/ratings.csv"))
pprint(better_df.dtypes)

# Cleaning data
# Common problems:
#   incorrect data types (from csv all are strings)
#   invalid rows (especially from manually entered data)
#   incomplete rows
#   badly chosen placeholders (like NA instead of null)
# Cleaning depends on the context
# Strict reporting environment (where every row of data counts)
# Can system cope with data that is 95% clean or 95% complete
# to drop bad rows:
prices = (spark.read.options(header="true", mode="DROPMALFORMED").csv('landing/prices.csv'))
# Spark's default way of handling missing data is to put in null
# Can fill that with what you want:
prices.fillna(25, subset=['quantity']).show()

# to conditionally replace bad values:
from pyspark.sql.functions import col, when
from datetime import date, timedelta
one_year_from_now = date.today().replace(year=date.today().year +1)
better_frame = employees.withColumns("end_date", 
    when(col("end_date") > one_year_from_now, None).otherwise(col("end_date")))

# Exercises:

# print("BEFORE")
ratings.show()

print("AFTER")
# Replace nulls with arbitrary value on column subset
ratings = ratings.fillna(4, subset=["comfort"])
ratings.show()  

from pyspark.sql.functions import col, when

# Add/relabel the column
categorized_ratings = ratings.withColumn(
    "comfort",
    # Express the condition in terms of column operations
    when(col("comfort") > 3, "sufficient").otherwise("insufficient"))

categorized_ratings.show()

# Tranform data
# Process: 
#   collect data
#   "massage data" : cleaning and business logic
#   derive insights
#  Common data transormations:
#   - filtering data
#   - selecting and renaming columns
#   - grouping rows and aggregation 
#   - joining multiple datasets
#   - ordering data (to prioritize)

# Filtering and ordering rows:
prices_in_belgium = prices.filter(col('countrycode') == 'BE').orderBy(col('date'))
# col function creates Column objects which can be compared to string literals and produce a bool Column
# orderBy - sorts values (like by date)
# Selecting and renaming columns:
prices.select(col("store"), col("brand").alias("brandname")).distinct()
# distinct gets rid of duplicates
# alias allows you to rename columns
# Grouping and aggregating with mean()
(prices.groupBy(col('brand')).mean('price')).show()
# Grouping and aggergating with agg() - to get both the avg and the count
(prices
  .groupBy(col('brand'))
  .agg(
    avg('price').alias('average_price'),
    count('brand').alias('number_of_items')
  )).show()
# Joining related data
# Executing a join with 2 foreign keys
ratings_with_prices = ratings.join(prices, ['brand', 'model'])

# Exercises:
from pyspark.sql.functions import col

# Select the columns and rename the "absorption_rate" column
result = ratings.select([col("brand"),
                       col("model"),
                       col("absorption_rate").alias('absorbency')])

# Show only unique values
result.distinct().show()

from pyspark.sql.functions import col, avg, stddev_samp, max as sfmax

aggregated = (purchased
              # Group rows by 'Country'
              .groupBy(col('Country'))
              .agg(
                # Calculate the average salary per group and rename
                avg('Salary').alias('average_salary'),
                # Calculate the standard deviation per group
                stddev_samp('Salary'),
                # Retain the highest salary per group and rename
                sfmax('Salary').alias('highest_salary')
              )
             )

aggregated.show()

# Running a pyspark program locally (just like regular Python)
# python my_pyspark_data_pipeline.py
# Need to have spark installed, access to referenced resources and configured classpath
# Normally, you'll use spark-submit (helper program)
# Sets up launch environment with cluster manager and the selected deploy mode
# Cluster manager makes cluster resources like RAM, CPU of different nodes available to other programs
# Deploy mode tells spark where the run the driver of Spark application (either on a 
#   dedicated master node or on a worker node)
# spark-submit also invokes the main class or main method
# spark-submit 
#   --master "local[*]" (often a URL of the cluster manager) 
#   --py-files PY_FILES (cs list of zip, egg, or py, copies dependencies over to the workers)
#   MAIN_PYTHON_FILE  (paht to the module to be run, contains code to trigger creation of SparkSession)
#   app_arguments
# zip files are common for the py-files
#   navigate to root folder of module
#   invoke the zip utility:
#   zip --recurse-paths dependencies.zip pydiaper
#   spark-submit --py-files dependencies.zip pydiaper/cleaning/clean_prices.py

# Pyspark Unit Tests
# Unit test are in the transformation layer (extract and load use other systems)
prices_with_ratings = spark.read.csv(...)
exchange_rates = spark.read.csv(...)
unit_prices_with_ratings = (prices_with_ratings
                          .join(...)
                          .withColumn(...))
# construct spark DF's in memory (dummy df's)
from pyspark.sql import Row
purchase = Row("price", "quantity", "product")
record = purchase(12.99, 1, "cake")
df = spark.createDataFrame((record,))

def link_with_exchange_rates(prices, rates):
  return prices.join(rates, ["currency", "date"])

def calculate_unit_price_in_euro(df):
  return df.withColumn("unit_price_in_euro", 
        col("price") / col("quantity") * col("exchange_rate_to_euro"))

def test_calculate_unit_price_in_euro():
  record = dict(price=10, quantity=5, exchange_rate_to_euro=2.)
  df = spark.createDateFrame([Row(**record)])
  result = calculate_unit_price_in_euro(df)

  expected_record = Row(**record, unit_price_in_euro=4.)
  expected = spark.createDataFrame([expected_record])

  assertDataFrameEqual(result, expected)

# Exercises:
from datetime import date
from pyspark.sql import Row

Record = Row("country", "utm_campaign", "airtime_in_minutes", "start_date", "end_date")

# Create a tuple of records
data = (
  Record("USA", "DiapersFirst", 28, date(2017, 1, 20), date(2017, 1, 27)),
  Record("Germany", "WindelKind", 31, date(2017, 1, 25), None),
  Record("India", "CloseToCloth", 32, date(2017, 1, 25), date(2017, 2, 2))
)

# Create a DataFrame from these records
frame = spark.createDataFrame(data)
frame.show()

# Continuous tests
# unittest and doctest are in standard library
# pytest is also great and lets you do a test suite
# pytest .  (runs all the tests)
# git can help automate by configuring hooks
# CI/CD pipeline (.circleci/config.yml) - this automates deployment
# checkout code
# install test and build requirements
# run tests with pytest
# package/build software

# Workflow management
# workflow - sequence of tasks that are scheduled or triggered by the occurrence of an event
# often schduled with cron
# cron reads config files known as crontab files
# Other workflow managers:
# Luigi (python based)
# Azkaban (java based)
# Airflow (python based)
# Airflow helps:
  # create and visualize complex workflows
  # monitor and logs workflows
  # scales horizontally
from airflow import DAG
my_dag = DAG(
    dag_id="publish_logs",
    schedule_interval="* * * * * ",
    start_date=datetime(2010, 1, 1)
)

from datetime import datetime
from airflow import DAG

reporting_dag = DAG(
    dag_id="publish_EMEA_sales_report", 
    # Insert the cron expression
    schedule_interval="0 7 * * 1",
    start_date=datetime(2019, 11, 24),
    default_args={"owner": "sales"}
)

# Specify direction using verbose method
prepare_crust.set_downstream(apply_tomato_sauce)

tasks_with_tomato_sauce_parent = [add_cheese, add_ham, add_olives, add_mushroom]
for task in tasks_with_tomato_sauce_parent:
    # Specify direction using verbose method on relevant task
    apply_tomato_sauce.set_downstream(task)

# Specify direction using bitshift operator
tasks_with_tomato_sauce_parent >> bake_pizza

# Specify direction using verbose method
bake_pizza.set_upstream(prepare_oven)

# BashOperator
from airflow.operators.bash_operator import BashOperator
bask_task=BashOperator(
  task_id="greet_world",
  dag=dag,
  bash_command='echo "hello, world!"'
)

python_task=PythonOperator(
  dag=dag,
  task_id="perform_magic",
  python_callable=my_function,
  op_kwargs={"snowflake": "*", "amount": 42}
)

# Running pyspark from Airflow:
spark_master=(
  "spark://"
  "spark_standalone_cluster_ip"
  ":7077")
command=(
  "spark-submit "
  "--master {master} "
  "--py-files package1.zip "
  "/path/to/app.py"
).format(master=spark_master)
BashOperator(bash_command=command)

# Must have spark binaries installed on the Airflow server
# Using ssh Operator (allows you to remote access a spark enabled cluster):
from airflow.contrib.operators.ssh_operator import SSHOperator
task = SSHOperator(
  task_id="ssh_spark_submit",
  dag=dag,
  command=command,
  ssh_conn_id="spark_master_ssh"
)
# can configure ssh_conn_id in Airflow user interface under Admin menu "connections"
# SparkSubmitOperator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
spark_task= SparkSubmitOperator(
  task_id="spark_submit_id",
  dag=dag,
  application="/path/to/app.py",
  py_files="package1.zip",
  conn_id="spark_default"
)

# Exercises:
# Create a DAG object
dag = DAG(
  dag_id='optimize_diaper_purchases',
  default_args={
    # Don't email on failure
    'email_on_failure': False,
    # Specify when tasks should have started earliest
    'start_date': datetime(2019, 6, 25)
  },
  # Run the DAG daily
  schedule_interval='@daily')

config = os.path.join(os.environ["AIRFLOW_HOME"], 
                      "scripts",
                      "configs", 
                      "data_lake.conf")

ingest = BashOperator(
  # Assign a descriptive id
  task_id="ingest_data", 
  # Complete the ingestion pipeline
  bash_command="tap-marketing-api | target-csv --config %s" % config,
  dag=dag)

# Import the operator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator

# Set the path for our files.
entry_point = os.path.join(os.environ["AIRFLOW_HOME"], "scripts", "clean_ratings.py")
dependency_path = os.path.join(os.environ["AIRFLOW_HOME"], "dependencies", "pydiaper.zip")

with DAG('data_pipeline', start_date=datetime(2019, 6, 25),
         schedule_interval='@daily') as dag:
  	# Define task clean, running a cleaning job.
    clean_data = SparkSubmitOperator(
        application=entry_point, 
        py_files=dependency_path,
        task_id='clean_data',
        conn_id='spark_default')

spark_args = {"py_files": dependency_path,
              "conn_id": "spark_default"}
# Define ingest, clean and transform job.
with dag:
    ingest = BashOperator(task_id='Ingest_data', bash_command='tap-marketing-api | target-csv --config %s' % config)
    clean = SparkSubmitOperator(application=clean_path, task_id='clean_data', **spark_args)
    insight = SparkSubmitOperator(application=transform_path, task_id='show_report', **spark_args)
    
    # set triggering sequence
    ingest >> clean >> insight

# Deploying Airflow
#  Installing and configuring Airflow (barebones)
# export AIRFLOW_HOME=~/airflow 
# pip install apache-airflow
# airflow initdb
# will create a subdirectory for all the log files, two configuration files, and a SQLite database
# need to configure Airflow to use the SequentialExecutor
# Production Airflow
# will have many more folders (dags, tests, connections, plugins, connection pools, variables )
# sometimes it is hard to debug without the web interface of Airflow, setting up a test like this can help
from airflow.models import DagBag
def test_dagbag_import():
  dagbag=DagBag()
  number_of_failures = len(dagbag.import_errors)
  assert number_of_failures == 0, "There should be no DAG failures. Got: %s" % dagbag.import_errors

# Exercises:
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

default_args = {
    #    "owner": "squad-a",
    "depends_on_past": False,
    "start_date": datetime(2019, 7, 5),
    "email": ["foo@bar.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "cleaning",
    default_args=default_args,
    user_defined_macros={"env": Variable.get("environment")},
    schedule_interval="0 5 */2 * *"
)


def say(what):
    print(what)


with dag:
    say_hello = BashOperator(task_id="say-hello", bash_command="echo Hello,")
    say_world = BashOperator(task_id="say-world", bash_command="echo World")
    shout = PythonOperator(task_id="shout",
                           python_callable=say,
                           op_kwargs={'what': '!'})

    say_hello >> say_world >> shout

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

default_args = {
    "owner": "squad-a",
    "depends_on_past": False,
    "start_date": datetime(2019, 7, 5),
    "email": ["foo@bar.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "cleaning",
    default_args=default_args,
    user_defined_macros={"env": Variable.get("environment")},
    schedule_interval="0 5 */2 * *"
)


def say(what):
    print(what)


with dag:
    say_hello = BashOperator(task_id="say-hello", bash_command="echo Hello,")
    say_world = BashOperator(task_id="say-world", bash_command="echo World")
    shout = PythonOperator(task_id="shout",
                           python_callable=say,
                           op_kwargs={'what': '!'})

    say_hello >> say_world >> shout
