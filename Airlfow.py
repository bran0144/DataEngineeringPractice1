# Workflow - set of steps to accomplish a tash (i.e. downloading files, copying data, filtering, writing to db)
# varying levels of complexity
# Airflow helps to program workflows including:
#   - creation
#   - scheduling
#   - monitoring
# Can implement programs from any language, but workflows are written in Python
# DAGs- Directed Acyclic Graphs (set of tasks and dependencies between them)
# can be accessed via code, CL, or web interface
# Other options: Luigi, SSIS, Bash scripting
# Includes detials like name, start date, owner, email alerting options
# Simple DAG definition:
etl_dag = DAG(
    dag_id='etl_pipeline',
    default_args={"start_date": "2020-01-08"}
)

# Running a workflow in Airlow
# in shell:
# run takes three arguments
# airflow run <dag_id> <task_id> <start_date>
# airflow run example-etl download-file 2020-10-01

# DAG
# Directed - inherent flow representing dependencies between components
    # dependencies provide context to the tools on how to order the running of components
# Acyclic - meaning it does not loop or repeat - individual compoenents are executed once per run
# Graph - actual set of components
# DAG's in Airflow:
# written in Python, case senstive in Python
# made of up components (tasks) to be executed (operators, sensors, etc)

from airflow.models import DAG
from datetime import datetime
default_arguments = {
    'owner': 'joe',
    'email': 'joe@google.com',
    'start_date': datetime(2020, 1, 20)
}
etl_dag = DAG('etl_workflow', default_args=default_arguments)

# DAG CL tool
# -h for help
# usually used for starting processes, manually run DAGS or tasks or review logging info

# Airflow operators
# Most common task
# Single task - usually run independently
# generally do not share information
# DummyOperator - can represent a task for troubleshooting or a task not yet implemented
# BashOperator - executes bash command or script
#   - requires 3 args - task_id, bash_command (can be raw command or script), dag
#   - runs in a temp dir that gets automatically cleaned up afterward
#   - can specify env variables for the command

from airflow.operators.bash_operator import BashOperator
example_task = BashOperator(task_id='bash_x',
                            bash_command='echo 1',
                            dag=dag)
bash_task=BashOperator(task_id='clean_addresses',
                        bash_command='cat addresses.txt | awk "NF==10" > clean.txt',
                        dag=dag)

# Gotchas
#  - operators are not guaranteed to run in the same location or environment.
#  - you may need to set up env variables (especially for BashOperator)
#  - can be difficult to run tasks with elevated privileges

# Exercises:
# Import the BashOperator
from airflow.operators.bash_operator import BashOperator

# Define the BashOperator 
cleanup = BashOperator(
    task_id='cleanup_task',
    # Define the bash_command
    bash_command='cleanup.sh',
    # Add the task to the dag
    dag=analytics_dag
)
# Define a second operator to run the `consolidate_data.sh` script
consolidate = BashOperator(
    task_id='consolidate_task',
    bash_command='consolidate_data.sh',
    dag=analytics_dag)

# Define a final operator to execute the `push_data.sh` script
push_data = BashOperator(
    task_id='pushdata_task',
    bash_command='push_data.sh',
    dag=analytics_dag)

# Tasks
# shortcut to refer to a given operator within a workflow
# usually assigned to a variable within Python code
# within airflow tools, the task is referred to by its task_id , not the variable name
# dependencies are usually present - define a given order of completion
# If dependencies are not defined, airflow does not guarantee order
# upstream tasks - must complete prior to any downstream tasks
# bitshift operators
#       >> upstream operator        can think of it as before
#       << downstream operator      after

task1 = BashOperator(...)
task2 = BashOperator(...)
task1 >> task2      #or you could do task2 << task1

# can chain dependencies
task1 >> task2 >> task3 >> task4
# you can mix operators within the same workflow
task1 >> task2 << task3

# Exercises:
# Define a new pull_sales task
pull_sales = BashOperator(
    task_id='pullsales_task',
    bash_command='wget https://salestracking/latestinfo?json',
    dag=analytics_dag
)

# Set pull_sales to run prior to cleanup
pull_sales >> cleanup

# Configure consolidate to run after cleanup
cleanup >> consolidate 

# Set push_data to run last
consolidate >> push_data

# Python Operator
# executes a python function/callable
# can pass in arguments to the python code
# supports positional and keyword arguments
# 

from airflow.operators.python_operator import PythonOperator
def printme():
    print("This goes in the logs")
python_task = PythonOperator(
    task_id='simple_print',
    python_callable=printme,
    dag= example_dag
)
# to keep kwargs organzied, use op_kwargs dictionary in your task
sleep_task = PythonOperator(
    task_id='sleep',
    python_callable=sleep,
    op_kwargs={'length_of_time': 5},
    dag=example_dag
)

# Many other operators to use
# EmailOperator
# sends email from within an Airflow task
# can contain HTML and attachments
# Airflow system must be configured with the email server details

from airflow.operators.email_operator import EmailOperator
email_task = EmailOperator(
    task_id='email_sales_report',
    to='sales@example.com',
    subject='Automated Sales Report',
    html_content='Attached is the latest sales report',
    files='lates_sales.xlsx',
    dag=example_dag
)

# Exercises:
def pull_file(URL, savepath):
    r = requests.get(URL)
    with open(savepath, 'wb') as f:
        f.write(r.content)   
    # Use the print method for logging
    print(f"File pulled from {URL} and saved to {savepath}")

from airflow.operators.python_operator import PythonOperator

# Create the task
pull_file_task = PythonOperator(
    task_id='pull_file',
    # Add the callable
    python_callable=pull_file,
    # Define the arguments
    op_kwargs={'URL':'http://dataserver/sales.json', 'savepath':'latestsales.json'},
    dag=process_sales_dag
)

# Add another Python task
parse_file_task = PythonOperator(
    task_id='parse_file',
    # Set the function to call
    python_callable=parse_file,
    # Add the arguments
    op_kwargs={'inputfile':'latestsales.json', 'outputfile':'parsedfile.json'},
    # Add the DAG
    dag=process_sales_dag
)

# Import the Operator
from airflow.operators.email_operator import EmailOperator

# Define the task
email_manager_task = EmailOperator(
    task_id='email_manager',
    to='manager@datacamp.com',
    subject='Latest sales JSON',
    html_content='Attached is the latest sales JSON file as requested.',
    files='parsedfile.json',
    dag=process_sales_dag
)

# Set the order of tasks
pull_file_task >> parse_file_task >> email_manager_task

# Airflow Scheduling

# DAG run - a specific instance of a workflow at a given time
# Can be run manually or with schedule_interval
# Each dag run maintains state for each workflow and the tasks within it
# state van be running, failed or success
# Dag Runs menu shows all the dag runs and their state

# Schedule details:
# start_date - date time to initialize run (usually python date time object) - not necessarily when they will actually run, just the first time they can be run
# end_date - optional for when to stop running new instances
# max_tries - optional number of attempts before failure
# schedule_interval - how often to run - between start_date and end_date
# can be set up with cron or with builtin presets

# Airflow scheduler presets:
# @hourly 0 * * * * 
# @daily 0 0  * * * 
# @weekly 0 0 * * 0

# Two special schedulers
# None - don’t ever schedule - should only be run manually
# @once - only scheduled once

# Won’t schedule until one interval past the start_date (so if every hour, it will wait until 1 hour after start date)


# Update the scheduling arguments as defined
default_args = {
  'owner': 'Engineering',
  'start_date': datetime(2019, 11, 1),
  'email': ['airflowresults@datacamp.com'],
  'email_on_failure': False,
  'email_on_retry': False,
  'retries': 3,
  'retry_delay': timedelta(minutes=20)
}

# dag = DAG('update_dataflows', default_args=default_args, schedule_interval='30 12 * * 3’)


# Airflow Sensor (type of operator)
# Sensor - operator that waits for a certain condition to be true
# examples: creation of a file, upload of data, web response
# Can define how often to check for the conditions to be true
# Are assigned to tasks just like other operators
# Need airflow.sensors.base_sensor_operator library
# Sensor arguments:
# mode- how to check for the condition
# mode=‘poke’ - the default, run repeatedly
# mode=‘reschedule’ - give up task slot and try again later
# poke_interval - how often to wait between checks (should be at least 1 minute to keep from overloading)
# timeout - how long to wait before failing task (should be significantly shorter than schedule interval)
# also includes normal operator attributes

# File Sensor:
# Part of airflow.contrib.sensors library
# checks for the existence of a file at a certain location
# can also check if any files exist within a directory

# from airflow.contrib.sensors.file_sensor import FileSensor

# file_sensor_task = FileSensor(task_id=‘file_sense’,
#     filepath=‘sales data.csv’,
#     poke_interval=300,
#     dag=sales_report_dag)

# init_sales_cleanup >> file_sensor_task >> generate_report

# Other sensors
# ExternalTaskSensor - wait for task in another DAG to complete (keeps dogs less complex)
# HttpSensor - request a web URL and check for content
# SqlSensor - checks for content

# Why use sensors?
# Uncertain when a condition will be true
# If failure not immediately desired
# To add task repetition without loops

# Airflow executors
# Executors run tasks
# different executors handle running the tasks differently
# Some examples: SequentialExecutor, LocalExecutor, CeleryExecutor
# SequentialExecutor:
    # default executor
    # runs one task at a time
    # useful for debugging
    # good for learning and testing, not recommended for production
    # race condition
# LocalExecutor:
    # runs entirely on a single system
    # treats tasks as processes and can run tasks concurrently (as many as allowed by system resources)
    # parallelism is defined by the user
    # unlimited parallelism or limited for a certain number of simultaneous tasks
    # good for a single productions Airflow system and can utilize all the resources of a given host system
# CeleryExecutor:
    # queing system in Python that allows multiple systems to communicate as a basic cluster
    # multiple airflow systems can be configured as workers for a given set of workflows/tasks
    # can add extra systems to better balance workflows
    # powerful choice for organizations with extensive workflows
# To find the executor from the CL:
# look at the airflow.cfg file
# can also find it by:
# airflow list_dags

# Exercises:
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from datetime import datetime

report_dag = DAG(
    dag_id = 'execute_report',
    schedule_interval = "0 0 * * *"
)

precheck = FileSensor(
    task_id='check_for_datafile',
    filepath='salesdata_ready.csv',
    start_date=datetime(2020,2,20),
    mode='reschedule',
    dag=report_dag
)

generate_report_task = BashOperator(
    task_id='generate_report',
    bash_command='generate_report.sh',
    start_date=datetime(2020,2,20),
    dag=report_dag
)

precheck >> generate_report_task

# Debugging and Troubleshooting
# Common issues:
    # Dag won't run on schedule - usually because scheduler isn't running
        # to fix from CL: airflow scheduler
        # executor does not have enough free slots to run tasks
            # change the executor type, add system resources, changing scheduling of your DAGs
    # dag won't load (either Dag not in webUI or no it list_dags)
        # verify Dag file is in correct folder
        # check dags folder via airflow.cfg (where airflow expects your dag files to be)
        # folder path must be an absolute path
    # syntax errors
        # airflow list_dags  - will output some debugging info
        # python3 <dagfile.py> - if there are errors you'll get an error message

# SLA's and reporting in Airflow
# SLA - service level agreement
# Within airflow - amount of time a task or DAG should require to run
# SLA miss - any time the task/DAG does not meet the expected timing
# If SLA is missed, an email is sent and a log is stored
# Can view in the webUI (under Browse-SLA Misses)
# Can configure sla argument on the task
# sla=timedelta(seconds=30)
# or is the default_args dictionary
# timedelta object
    # in the datetime library
    # from datetime import delta
    # takes arguments of days, seconds, minutes, hours, weeks
# General reporting
# can use email alerting built in to airflow
# can send on success/failure/error
#  handled through keys in the default_args dictionary
# required argument is the list of emails assigned to the email key
default_args={
    'email': ['example@datacamp.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'email_on_success': True,
}

# Exercises:
# Import the timedelta object
from datetime import timedelta

# Create the dictionary entry
default_args = {
  'start_date': datetime(2020, 2, 20),
  'sla': timedelta(minutes=30)
}

# Add to the DAG
test_dag = DAG('test_workflow', default_args=default_args, schedule_interval='@None')

test_dag = DAG('test_workflow', start_date=datetime(2020,2,20), schedule_interval='@None')

# Create the task with the SLA
task1 = BashOperator(task_id='first_task',
                     sla=timedelta(hours=3),
                     bash_command='initialize_data.sh',
                     dag=test_dag)


# Define the email task
email_report = EmailOperator(
        task_id='email_report',
        to='airflow@datacamp.com',
        subject='Airflow Monthly Report',
        html_content="""Attached is your monthly workflow report - please refer to it for more detail""",
        files=['monthly_report.pdf'],
        dag=report_dag
)

# Set the email task to run after the report is generated
email_report << generate_report

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from datetime import datetime

default_args={
    'email': ['airflowalerts@datacamp.com', 'airflowadmin@datacamp.com'],
    'email_on_failure': True,
    'email_on_success': True
}
report_dag = DAG(
    dag_id = 'execute_report',
    schedule_interval = "0 0 * * *",
    default_args=default_args
)

precheck = FileSensor(
    task_id='check_for_datafile',
    filepath='salesdata_ready.csv',
    start_date=datetime(2020,2,20),
    mode='reschedule',
    dag=report_dag)

generate_report_task = BashOperator(
    task_id='generate_report',
    bash_command='generate_report.sh',
    start_date=datetime(2020,2,20),
    dag=report_dag
)

precheck >> generate_report_task

# Templates
# allow substitution of information during a DAG run
# provide added flexibility when defining tasks
# Created using Jinja templating language
templated_command="""
    echo "Reading {{ params.filename }}"
"""
t1 = BashOperator(task_id='template_task',
    bash_command=templated_command,
    params={filename: 'file1.txt'},
    dag=example_dag)
t2 = BashOperator(task_id='template_task',
    bash_command=templated_command,
    params={filename: 'file2.txt'},
    dag=example_dag)

# Exercises:
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
  'start_date': datetime(2020, 4, 15),
}

cleandata_dag = DAG('cleandata',
                    default_args=default_args,
                    schedule_interval='@daily')

# Create a templated command to execute
# 'bash cleandata.sh datestring'
templated_command = """
bash cleandata.sh {{ ds_nodash }}
"""

# Modify clean_task to use the templated command
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          dag=cleandata_dag)

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
  'start_date': datetime(2020, 4, 15),
}

cleandata_dag = DAG('cleandata',
                    default_args=default_args,
                    schedule_interval='@daily')

# Modify the templated command to handle a
# second argument called filename.
templated_command = """
  bash cleandata.sh {{ ds_nodash }} {{ params.filename }}
"""

# Modify clean_task to pass the new argument
clean_task = BashOperator(task_id='cleandata_task',
                          bash_command=templated_command,
                          params={'filename': 'salesdata.txt'},
                          dag=cleandata_dag)

# Create a new BashOperator clean_task2
clean_task2 = BashOperator(task_id='cleandata_task2',
                           bash_command=templated_command,
                           params={'filename': 'supportdata.txt'},
                           dag=cleandata_dag)
                           
# Set the operator dependencies
clean_task >> clean_task2

# More templates
