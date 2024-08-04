import os
import datetime
import matplotlib.pyplot as plt
import shap
import numpy as np
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def create_folder(name):
    date_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"models_{name}_{date_now}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

mlflow.set_tracking_uri('http://mlflow-server:5001')
mlflow.set_experiment('kb_project')

@transformer
def transform(data, *args, **kwargs):
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    catboost_model = CatBoostClassifier(verbose=0, depth=6, iterations=300, learning_rate=0.3)

    with mlflow.start_run() as run:
        catboost_model.fit(X_train, y_train)
        y_pred_catboost = catboost_model.predict(X_test)

        folder_name = create_folder('catboost')

        model_path = os.path.join(folder_name, "catboost_model.cbm")
        catboost_model.save_model(model_path)

        accuracy = round(accuracy_score(y_test, y_pred_catboost), 4)
        precision = round(precision_score(y_test, y_pred_catboost), 4)
        recall = round(recall_score(y_test, y_pred_catboost), 4)
        f1 = round(f1_score(y_test, y_pred_catboost), 4)

        result_string = f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}'
        print(result_string)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        metrics_path = os.path.join(folder_name, 'metrics.txt')
        with open(metrics_path, 'w') as file:
            file.write(result_string)

        cm_catboost = confusion_matrix(y_test, y_pred_catboost)
        disp_catboost = ConfusionMatrixDisplay(confusion_matrix=cm_catboost, display_labels=catboost_model.classes_)
        disp_catboost.plot()
        confusion_matrix_path = os.path.join(folder_name, "confusion_matrix.png")
        plt.title("CatBoost Confusion Matrix")
        plt.savefig(confusion_matrix_path)
        plt.close()

        feature_importances = catboost_model.get_feature_importance()
        feature_importance_path = os.path.join(folder_name, "feature_importance.txt")
        with open(feature_importance_path, "w") as f:
            for feature_name, importance in zip(X.columns, feature_importances):
                f.write(f"{feature_name}: {importance}\n")

        explainer = shap.TreeExplainer(catboost_model)
        shap_values = explainer.shap_values(X_test)
        shap_values_path = os.path.join(folder_name, "shap_values.npy")
        np.save(shap_values_path, shap_values)

        shap_summary_plot_path = os.path.join(folder_name, "shap_summary_plot.png")
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(shap_summary_plot_path)
        plt.close()

        mlflow.catboost.log_model(catboost_model, "catboost_model")
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(feature_importance_path)
        mlflow.log_artifact(shap_values_path)
        mlflow.log_artifact(shap_summary_plot_path)

        mlflow.log_params({
            "depth": 6,
            "iterations": 300,
            "learning_rate": 0.3
        })

    return f'Model saved in {folder_name}'

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
