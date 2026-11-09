from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor


def train_model_and_predict(train_file: str, test_file: str) -> np.ndarray:
    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    y_train = df_train["awards"].copy()
    df_train = df_train.drop(columns=["awards"])

    def convert_to_string(x):
        if isinstance(x, list):
            items = [str(item) if item is not None else '' for item in x]
            return ' '.join(items)
        elif x is None:
            return ''
        else:
            return str(x)

    for col in df_train.columns:
        if df_train[col].dtype == 'object':
            df_train[col] = df_train[col].apply(convert_to_string)
            if col in df_test.columns:
                df_test[col] = df_test[col].apply(convert_to_string)

    text_columns = ['genres', 'directors', 'filming_locations', 'keywords']

    for col in text_columns:
        if col in df_train.columns and col in df_test.columns:
            tfidf = TfidfVectorizer(max_features=50, stop_words='english')
            train_tfidf = tfidf.fit_transform(df_train[col].fillna('')).toarray()
            test_tfidf = tfidf.transform(df_test[col].fillna('')).toarray()


            tfidf_cols = [f'{col}_tfidf_{i}' for i in range(train_tfidf.shape[1])]
            train_tfidf_df = pd.DataFrame(train_tfidf, columns=tfidf_cols, index=df_train.index)
            test_tfidf_df = pd.DataFrame(test_tfidf, columns=tfidf_cols, index=df_test.index)

            df_train = pd.concat([df_train, train_tfidf_df], axis=1)
            df_test = pd.concat([df_test, test_tfidf_df], axis=1)

            df_train = df_train.drop(columns=[col])
            df_test = df_test.drop(columns=[col])

    categorical_columns = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender', 'actor_3_gender']

    for col in categorical_columns:
        if col in df_train.columns:
            le = LabelEncoder()
            if col in df_test.columns:
                combined = pd.concat([df_train[col].fillna('unknown'), df_test[col].fillna('unknown')])
            else:
                combined = df_train[col].fillna('unknown')
            le.fit(combined.astype(str))
            df_train[col] = le.transform(df_train[col].fillna('unknown').astype(str))
            if col in df_test.columns:
                df_test[col] = le.transform(df_test[col].fillna('unknown').astype(str))

    numeric_columns = df_train.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_train[col].isnull().any():
            median_val = df_train[col].median()
            df_train[col].fillna(median_val, inplace=True)
            if col in df_test.columns:
                df_test[col].fillna(median_val, inplace=True)

    for col in df_test.columns:
        if col not in df_train.columns:
            df_test = df_test.drop(columns=[col])

    for col in df_train.columns:
        if col not in df_test.columns:
            df_test[col] = 0

    df_test = df_test[df_train.columns]

    model = LGBMRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

    model.fit(df_train, y_train)
    predictions = model.predict(df_test)

    return predictions
