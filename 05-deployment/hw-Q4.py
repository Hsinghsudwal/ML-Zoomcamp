import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify

with open ('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

with open ('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('churn')

def predict_single(client, dv, model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)


    #  = {"job": "student", "duration": 280, "poutcome": "failure"}
