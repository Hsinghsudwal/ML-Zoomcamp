import pickle
from flask import Flask, request, jsonify

with open('pipeline_v1.bin', 'rb') as f_in: 
    dv, model = pickle.load(f_in)

app = Flask('churn')

def predict_single(inputs, dv, model):
    X = dv.transform([inputs])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

@app.route('/predict', methods=['POST'])
def predict():
    inputs = request.get_json()
    prediction = predict_single(inputs, dv, model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)
