import time
from datetime import datetime
from airflow.models.dag import DAG
from airflow.decorators import task
from airflow.utils.task_group import TaskGroup
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.hooks.base import BaseHook
import pandas as pd
from sqlalchemy import create_engine

#Function to extract the data from mysql source

@task()
def get_src_tables():
    hook = MySqlHook(mysql_conn_id="mysql_conn") # connection for airflow
    sql = """ SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'coronary' AND table_name = 'dataset';"""
    df = hook.get_pandas_df(sql) # built-in function to get pandas data frame
    print(df)
    tbl_dict = df.to_dict('dict') # When data is serialized, it is converted into a format that can be stored or transmitted across the tasks. So, used dictionaries
    return tbl_dict

# Storing the data in postgres
@task()
def load_src_data(tbl_dict: dict):
    conn = BaseHook.get_connection('postgres_conn') # connection details for postgres in airflow
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}') # sqlalchemy module to create connection
    all_table_name = [] # array to store table names
    start_time = time.time() # get current time
    #Iterate the table content in dictionaries
    for k, v in tbl_dict['table_name'].items():
        all_table_name.append(v)
        rows_imported = 0
        sql = f'select * from {v}' # Dynamic sql query for iterating table names {v}
        hook = MySqlHook(mysql_conn_id = "mysql_conn")
        df = hook.get_pandas_df(sql)
        print(f'importing rows {rows_imported} to {rows_imported + len(df)}... for table {v}') # To monitor how many rows importing from which table
        df.to_sql(f'src_{v}', engine, if_exists= 'replace', index= False) # Persist the data in postgres indicating as source table
        rows_imported += len(df)
        print(f'Done. {str(round(time.time() - start_time, 2))} total seconds elapsed')
    print("Data imported successful")
    return all_table_name

# Function to make some transformations on tables
@task()
def transform_cad_dataset():
    conn = BaseHook.get_connection('postgres_conn')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')
    pdf = pd.read_sql_query('SELECT * FROM itc.dataset', engine)
    #drop columns
    revised = pdf[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class']]
    revised.drop('class', axis=1,inplace=True)
    revised.drop('sex', axis=1,inplace=True)
    revised.drop('cp', axis=1,inplace=True)
    revised.drop('exang', axis=1,inplace=True)
    revised.rename(columns={'fbs':'Diabetes'}) # renaming column names
    revised.rename(columns={'chol':'cholestrol'})
    revised.rename(columns={'thal':'thalassemia'})
    revised.to_sql(f'stg_dataset', engine, if_exists='replace', index=False)# Move transformed data in staging table
    return {"table(s) processed": "Data imported successful"}

# Function for loading the transformed data into the PostgresSQL
@task()
def data_model():
    conn = BaseHook.get_connection('postgres')
    engine = create_engine(f'postgresql://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}')
    pc = pd.read_sql_query('SELECT * FROM itc."dataset"', engine)
    pc.to_sql(f'transformed_dataset', engine, if_exists='replace', index=False) # New table with transformed data
    return {"table(s) processed": "Data imported successful"}

# Assining the Task Group and dependencies that can airflow make to run the jobs
with DAG(dag_id="my_etl_dag", schedule="0 22 * * *", start_date=datetime(2023, 4, 15), catchup=False, tags=["data_model"]) as dag:

    with TaskGroup("extract_dataset_load", tooltip="Extract and Load source data") as extract_load_src:
        src_dataset_tbls = get_src_tables()
        load_dataset = load_src_data(src_dataset_tbls)
        load_dataset

    with TaskGroup("transform_src_dataset", tooltip="Transform and stage data") as transform_src_dataset:
        transform_src = transform_cad_dataset()
        transform_src

    with TaskGroup("load_data_model", tooltip="final data model") as load_data_model:
        final_data = data_model()
        final_data

    extract_load_src >> transform_src_dataset >> load_data_model # Dependencies
