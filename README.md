**Health Risk ML
Small example project to train a simple classifier on health data and produce predictions.**

**Contents**

app.py — CLI entrypoint (train and predict commands).

preprocess.py — loading, preprocessing, and train/test split helpers.

train_model.py — training script that saves a model/model.pkl and preprocessor.

predict.py — load model + preprocessor to predict on new CSVs.

metrics.py — helper to compute common metrics.

health_data.csv — small example dataset.


**Setup**

Create a Python environment and install dependencies:

If your system uses py or a different python executable, use that instead.

**Usage**

Train a model (writes model/model.pkl and model/preprocessor.joblib):

Predict on a CSV with the same feature columns (no label column required):


**Notes**

The code uses a RandomForestClassifier and simple numeric imputation + scaling.
Adjust preprocess.py if your dataset has categorical features or different column names.
