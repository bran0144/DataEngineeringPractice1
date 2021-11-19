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