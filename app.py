import argparse
from model.train_model import train
from model.predict import predict


def main():
	parser = argparse.ArgumentParser(description='Health risk ML helper')
	sub = parser.add_subparsers(dest='cmd')

	t = sub.add_parser('train')
	t.add_argument('--data', default='data/health_data.csv')
	t.add_argument('--label', default='target')

	p = sub.add_parser('predict')
	p.add_argument('--input', required=True)
	p.add_argument('--model', default='model/model.pkl')
	p.add_argument('--preproc', default='model/preprocessor.joblib')

	args = parser.parse_args()
	if args.cmd == 'train':
		train(data_path=args.data, label_col=args.label)
	elif args.cmd == 'predict':
		preds = predict(args.input, model_path=args.model, preproc_path=args.preproc)
		print(preds.tolist() if hasattr(preds, 'tolist') else preds)
	else:
		parser.print_help()


if __name__ == '__main__':
	main()

