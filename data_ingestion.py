# In data_ingestion.py

import logging
from pyspark.sql import SparkSession

# Initializes a SparkSession with necessary configurations.
def create_spark_session():
    
    spark = SparkSession.builder \
        .appName("DataIngestion") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth.Subject=ALL-UNNAMED") \
        .getOrCreate()
    return spark

# Ingests the raw data files (train.csv and test.csv) and performs initial cleaning.
def ingest_data(spark):
    logging.info("Starting data ingestion.")
    try:
        # Load the two separate datasets
        train_df = spark.read.csv("data/train.csv", header=True, inferSchema=True)
        test_df = spark.read.csv("data/test.csv", header=True, inferSchema=True)
        logging.info("Data ingestion successful.")

        # Initial cleaning: dropping columns with many missing values
        train_df = train_df.drop('Cabin', 'Ticket').fillna({'Embarked': 'S'})
        test_df = test_df.drop('Cabin', 'Ticket').fillna({'Embarked': 'S'})
        
        # We will split the train_df into train and validate sets in the main pipeline script.
        
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = create_spark_session()
    train_data, test_data = ingest_data(spark)
    spark.stop()