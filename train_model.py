from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


MODEL_PATH = Path(__file__).parent / "titanic_model.joblib"

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

TARGET = "survived"


def ensure_model():

    if MODEL_PATH.exists():
        return

    data = pd.read_csv(DATA_URL)

    # Fix column names
    data = data.rename(columns={
        "pclass": "Pclass",
        "sex": "Sex",
        "age": "Age",
        "sibsp": "SibSp",
        "parch": "Parch",
        "fare": "Fare",
        "embarked": "Embarked"
    })

    data = data[FEATURES + [TARGET]]

    data = data.dropna(subset=[TARGET])

    numeric_features = [
        "Pclass", "Age", "SibSp", "Parch", "Fare"
    ]

    categorical_features = ["Sex", "Embarked"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        ))
    ])

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved successfully!")


if __name__ == "__main__":
    ensure_model()
