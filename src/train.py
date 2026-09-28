import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("wine-classifier")

BASELINE_ACCURACY = 0.90  # Compare against this each run

def main():
    df = pd.read_csv("data/processed.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        # Hyperparameter Tuning
        param_grid = {"n_estimators": [50, 100], "max_depth": [4, 8, None]}
        search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)

        # Performance Evaluation
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy:.2%} (baseline: {BASELINE_ACCURACY:.2%})")

        # Model Versioning — artifacts + metadata via MLflow, plus a plain file for the deploy step
        mlflow.sklearn.log_model(best_model, "model")
        joblib.dump(best_model, "src/model.joblib")

        if accuracy < BASELINE_ACCURACY:
            raise SystemExit(f"Accuracy {accuracy:.2%} below baseline {BASELINE_ACCURACY:.2%} — failing build")

if __name__ == "__main__":
    main()