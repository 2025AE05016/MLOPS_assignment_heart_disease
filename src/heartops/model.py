from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import NUMERIC, CATEGORICAL


def build_pipeline(classifier):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    category = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return Pipeline([("features", ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("category", category, CATEGORICAL)])),
        ("classifier", classifier)])
