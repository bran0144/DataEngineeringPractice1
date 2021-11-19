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

