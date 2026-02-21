from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# model load
with open('bangalore_home_prices_model (1).pkl', 'rb') as f:
    model = pickle.load(f)

# ✅ extract locations for dropdown
locations = model.named_steps['columntransformer'] \
    .transformers_[0][1] \
    .categories_[0]

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    input_df = pd.DataFrame(
        [[location, sqft, bath, bhk]],
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )

    price = model.predict(input_df)[0]

    return render_template(
        'index.html',
        prediction_text=f"Estimated Price: ₹ {round(price, 2)} Lakhs",
        locations=locations
    )

if __name__ == '__main__':
    app.run(debug=True)
