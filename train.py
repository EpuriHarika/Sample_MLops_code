import logging
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)

def train():
    logging.info("Loading dataset...")
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    logging.info("Training model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    logging.info(f"Model accuracy: {acc}")

    joblib.dump(model, "model.pkl")
    logging.info("Model saved as model.pkl")

if __name__ == "__main__":
    train()