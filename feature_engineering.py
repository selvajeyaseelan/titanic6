# In feature_engineering.py
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, when, split, lit
import logging

# Performs feature engineering by adding a 'Title' column.
def add_title_feature(df):
    
    logging.info("Starting feature engineering to add 'Title' column.")
    try:
        df_with_title = df.withColumn("Title", when(split(col("Name"), r"\s+")[1].isin("Mr.", "Mrs.", "Miss.", "Master."), split(col("Name"), r"\s+")[1]).otherwise(lit("Other")))
        logging.info("'Title' feature added successfully.")
        return df_with_title
    except Exception as e:
        logging.error(f"Error during 'Title' feature engineering: {e}")
        raise

# Assembles all feature columns into a single vector.
def assemble_features(df):
    
    logging.info("Assembling features.")
    assembler = VectorAssembler(
        inputCols=[
            "Pclass", "Age_imputed", "SibSp", "Parch", "Fare_imputed",
            "SexVec", "EmbarkedVec", "TitleVec"
        ],
        outputCol="features"
    )
    return assembler.transform(df)