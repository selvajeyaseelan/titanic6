# Import libraries
import logging
import os
from pyspark.sql import SparkSession
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, split, lit
from pyspark.ml import PipelineModel


# Define local paths and model names
PREPROCESSING_PIPELINE_PATH = "./preprocessing_pipeline_model"
REGISTERED_MODEL_NAME = "TitanicBestModel"
MLFLOW_TRACKING_URI = "http://172.18.249.244:5000" # ML Flow server tracking user interface

# Import functions from your other modules
from data_ingestion import ingest_data
from preprocessing import build_preprocessing_pipeline
from model_building_lr import build_model, build_param_grid
from feature_engineering import add_title_feature

# Initializes and returns a SparkSession with necessary configurations.
def create_spark_session():
    
    spark = SparkSession.builder \
        .appName("TitanicMLOpsPipeline") \
        .config("spark.driver.extraJavaOptions", "--add-opens=java.base/javax.security.auth.Subject=ALL-UNNAMED") \
        .getOrCreate()
    return spark

# Main function to run the complete MLOps pipeline.
def main():
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the MLOps pipeline run.")

    # Set up MLflow tracking URI
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    logging.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

    try:
        spark = create_spark_session()
        logging.info("Spark session created successfully.")
    except Exception as e:
        logging.error(f"Failed to create Spark session: {e}")
        return
    
    # DVC Integration
    try:
        logging.info("Pulling data from DVC.")
        os.system("dvc pull")
        logging.info("Data pull from DVC completed.")
    except Exception as e:
        logging.error(f"Failed to pull data from DVC: {e}")
        spark.stop()
        return

    # Step 1: Data Ingestion and Initial Cleaning
    try:
        logging.info("Starting data ingestion.")
        train_df = spark.read.csv("data/train.csv", header=True, inferSchema=True)
        test_df = spark.read.csv("data/test.csv", header=True, inferSchema=True)
        
        train_df = train_df.drop('Cabin', 'Ticket').fillna({'Embarked': 'S'})
        test_df = test_df.drop('Cabin', 'Ticket').fillna({'Embarked': 'S'})
        
        logging.info("Data ingested and initially cleaned.")
        print("--- Ingested Training Data (First 5 rows) ---")
        train_df.show(5)
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        spark.stop()
        return

    # Step 2: Data Preprocessing and Feature Engineering
    try:
        logging.info("Starting data preprocessing and feature engineering.")
        
        train_df = add_title_feature(train_df)
        test_df = add_title_feature(test_df)

        pipeline = build_preprocessing_pipeline()
        pipeline_model = pipeline.fit(train_df)
        
        pipeline_model.write().overwrite().save(PREPROCESSING_PIPELINE_PATH)
        logging.info(f"Fitted preprocessing pipeline model saved to {PREPROCESSING_PIPELINE_PATH}")

        processed_train_df = pipeline_model.transform(train_df)
        processed_test_df = pipeline_model.transform(test_df)
        
        logging.info("Data preprocessing and feature engineering completed.")
        print("Processed Training Data (First 5 rows with 'features' column)")
        processed_train_df.select("Survived", "features").show(5, truncate=False)
    except Exception as e:
        logging.error(f"Preprocessing/Feature Engineering failed: {e}")
        spark.stop()
        return

    # Step 3: Train and Evaluate Multiple Models
    logging.info("Starting multi-model training and evaluation.")

    (training_data, validation_data) = processed_train_df.randomSplit([0.8, 0.2], seed=42)
    models_to_train = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosted_trees"]
    
    best_overall_auc = -1
    best_overall_model = None
    best_overall_model_name = ""

    for model_name in models_to_train:
        with mlflow.start_run(run_name=f"Training_{model_name}"):
            try:
                print(f"\n Starting training for: {model_name}:")
                base_model = build_model(model_name)
                param_grid = build_param_grid(model_name)
                
                mlflow.log_param("model_name", model_name)

                evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
                cross_validator = CrossValidator(
                    estimator=base_model,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=5
                )

                cv_model = cross_validator.fit(training_data)
                best_model = cv_model.bestModel
                
                logging.info(f"Hyperparameter tuning for {model_name} complete.")

                predictions = best_model.transform(validation_data)
                auc = evaluator.evaluate(predictions)
                multi_evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction")
                accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
                f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
                precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
                recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
                
                ###
                mlflow.log_metric("validation_auc", auc)
                logging.info(f"Final Model {model_name} AUC on validation set: {auc}")
                print(f"Model {model_name} Evaluation")
                print(f"Validation AUC: {auc}")
                print(" Example Predictions (First 5 rows)")
                predictions.select("Survived", "prediction", "probability").show(5)

                # EVALUATION ON TRAINING DATA

                print(f"\n--- Model {model_name} Training Metrics ---")
                train_predictions = best_model.transform(training_data)
                train_auc = evaluator.evaluate(train_predictions)
                multi_evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction")
                train_accuracy = multi_evaluator.evaluate(train_predictions, {multi_evaluator.metricName: "accuracy"})
                train_f1 = multi_evaluator.evaluate(train_predictions, {multi_evaluator.metricName: "f1"})

                print('---------------------------------------')
                print("Training Scores:")
                print(f"Training AUC: {train_auc}")
                print(f"Training Accuracy: {train_accuracy}")
                print(f"Training F1-Score: {train_f1}")

                print('---------------------------------------')
                print("Testing Scores:")
                print(f"Validation AUC: {auc}")
                print(f"Validation Accuracy: {accuracy}")
                print(f"Validation F1-Score: {f1}")
                print(f"Validation Precision: {precision}")
                print(f"Validation Recall: {recall}")
                print('---------------------------------------')

                if auc > best_overall_auc:
                    best_overall_auc = auc
                    best_overall_model = best_model
                    best_overall_model_name = model_name
                    
            except Exception as e:
                logging.error(f"Model training/tuning for {model_name} failed: {e}")
                mlflow.end_run("FAILED")
    
    # After the loop, log the single best model to MLflow
    if best_overall_model:
        with mlflow.start_run(run_name="Final_Best_Model_Eval"):
            logging.info(f"The best overall model is: {best_overall_model_name} with AUC: {best_overall_auc}")
            print('---------------------------------------')
            print(f"Best Model Selected: ")
            print(f"Best Model: {best_overall_model_name}")
            print(f"Best Overall AUC: {best_overall_auc}")
            print('---------------------------------------')
            mlflow.log_param("best_overall_model_name", best_overall_model_name)
            mlflow.log_metric("best_overall_auc", best_overall_auc)
            print('---------------------------------------')
            
            # Save the final preprocessor and model together for deployment
            final_pipeline = PipelineModel(stages=[
                pipeline_model.stages[0],
                pipeline_model.stages[1],
                pipeline_model.stages[2],
                pipeline_model.stages[3],
                pipeline_model.stages[4],
                pipeline_model.stages[5],
                pipeline_model.stages[6],
                pipeline_model.stages[7],
                best_overall_model
            ])

            # Log the complete pipeline (preprocessor + model) to MLflow
            mlflow.spark.log_model(
                spark_model=final_pipeline,
                artifact_path="full-pipeline-model",
                registered_model_name=REGISTERED_MODEL_NAME
            )
            logging.info("Full pipeline model logged to MLflow.")

            # MLflow Model Registry Stage Transition 
            client = MlflowClient()
            latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0]
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=latest_version.version,
                stage="Staging"
            )
            logging.info(f"Model version {latest_version.version} transitioned to Staging.")

    # Step 4: Stop Spark session
    spark.stop()
    logging.info("Pipeline run finished.")

if __name__ == "__main__":
    main()