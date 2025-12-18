import joblib
import pandas as pd
from pathlib import Path
from preprocessing.preprocess import load_data, load_preprocessor


def predict(input_path, model_path='model/model.pkl', preproc_path='model/preprocessor.joblib'):
	model = joblib.load(model_path)
	preproc = load_preprocessor(preproc_path)
	df = load_data(input_path)
	X = df.copy()
	num_cols = preproc.get('num_cols', [])
	if num_cols:
		X[num_cols] = preproc['imputer'].transform(X[num_cols])
		if preproc['scaler'] is not None:
			X[num_cols] = preproc['scaler'].transform(X[num_cols])
	preds = model.predict(X)
	return preds


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', required=True)
	parser.add_argument('--model', default='model/model.pkl')
	parser.add_argument('--preproc', default='model/preprocessor.joblib')
	args = parser.parse_args()
	results = predict(args.input, model_path=args.model, preproc_path=args.preproc)
	print(results.tolist() if hasattr(results, 'tolist') else results)

