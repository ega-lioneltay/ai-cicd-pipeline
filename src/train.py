from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("wine-classifier")

BASELINE_ACCURACY = 0.90


def main():
    df = pd.read_csv("data/processed.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        param_grid = {"n_estimators": [50, 100], "max_depth": [4, 8, None]}
        search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)

        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy:.2%} (baseline: {BASELINE_ACCURACY:.2%})")

        if accuracy < BASELINE_ACCURACY:
            raise SystemExit(
                f"Accuracy {accuracy:.2%} below baseline {BASELINE_ACCURACY:.2%} — failing build"
            )

        # Log model with trusted skops types and address the 'name' deprecation warning
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        Path("src").mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, "src/model.joblib")


if __name__ == "__main__":
    main()