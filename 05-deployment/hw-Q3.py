import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

with open ('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open ('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

client = {"job": "management", "duration": 400, "poutcome": "success"}

X= dv.transform([client])
y_pred=model.predict_proba(X)[0,1]

print('client:', client, 'prediction:', y_pred)