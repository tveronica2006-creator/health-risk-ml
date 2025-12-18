from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(path):
	return pd.read_csv(path)


def preprocess(df, label_col='target'):
	if label_col not in df.columns:
		raise ValueError(f"Label column '{label_col}' not found in dataframe")
	X = df.drop(columns=[label_col])
	y = df[label_col]
	num_cols = X.select_dtypes(include=['number']).columns.tolist()
	imputer = SimpleImputer(strategy='median')
	if num_cols:
		X[num_cols] = imputer.fit_transform(X[num_cols])
		scaler = StandardScaler()
		X[num_cols] = scaler.fit_transform(X[num_cols])
	else:
		scaler = None

	preprocessor = {
		'imputer': imputer,
		'scaler': scaler,
		'num_cols': num_cols,
	}
	return X, y, preprocessor


def split(X, y, test_size=0.2, random_state=42):
	return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_preprocessor(preproc, path):
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(preproc, path)


def load_preprocessor(path):
	return joblib.load(path)

