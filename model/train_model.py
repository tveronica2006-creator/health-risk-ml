import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocessing.preprocess import load_data, preprocess, split, save_preprocessor


def train(data_path='data/health_data.csv', label_col='target', model_out='model/model.pkl', preproc_out='model/preprocessor.joblib'):
	df = load_data(data_path)
	X, y, preproc = preprocess(df, label_col=label_col)
	X_train, X_test, y_train, y_test = split(X, y)
	clf = RandomForestClassifier(n_estimators=100, random_state=42)
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)
	acc = accuracy_score(y_test, preds)
	Path(model_out).parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(clf, model_out)
	save_preprocessor(preproc, preproc_out)
	print(f"Trained model saved to {model_out}; preprocessor saved to {preproc_out}; test accuracy {acc:.4f}")
	return clf, preproc


if __name__ == '__main__':
	train()

