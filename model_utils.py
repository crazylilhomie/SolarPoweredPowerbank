import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def load_data(file):
    return pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

def preprocess(df):
    df = df.copy()
    df = df.dropna(axis=1, how="all")

    target = [c for c in df.columns if "Target" in c or "Buy" in c or "purchase" in c.lower()][0]

    X = df.drop(columns=[target])
    y = df[target]

    encoders = {}
    for col in X.select_dtypes(include="object"):
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        encoders[col] = enc

    return X, y, encoders

def train_models(df):
    X, y, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = accuracy_score(y_test, preds)

    return results

def predict_new(data_dict, df):
    X, y, encoders = preprocess(df)

    input_df = pd.DataFrame([data_dict])

    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])

    model = RandomForestClassifier()
    model.fit(X, y)
    return model.predict(input_df)[0]
