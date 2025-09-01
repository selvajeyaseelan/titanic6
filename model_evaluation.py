from pyspark.ml.evaluation import BinaryClassificationEvaluator
import logging
import mlflow

# Evaluates a trained model and logs metrics to MLflow.
def evaluate_model(model, validation_data):
    
    logging.info("Starting model evaluation.")
    try:
        predictions = model.transform(validation_data)
        evaluator = BinaryClassificationEvaluator(
            labelCol="Survived",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(predictions)
        
        # Log the evaluation metric
        mlflow.log_metric("AUC", auc)
        logging.info(f"Model evaluated. AUC: {auc}")
        
        return auc
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise