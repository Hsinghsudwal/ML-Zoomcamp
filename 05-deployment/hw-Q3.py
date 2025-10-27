import pickle

with open('pipeline_v1.bin', 'rb') as f_in: 
    dv, model = pickle.load(f_in)

inputs= {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = dv.transform([inputs])
y_pred=model.predict_proba(X)[0,1]

print('input:', inputs, 'prediction:', y_pred)