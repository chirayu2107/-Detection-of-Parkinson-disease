from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)

# Load the trained model and scaler
model = SVC(kernel='linear', C=1.0)
scaler = StandardScaler()

# Load the Parkinson's dataset and preprocess it
parkinsons_data = pd.read_csv(r'C:\Users\Yash Singh\OneDrive\Desktop\Yash\SEM 6\Foundation_of_datascience\archive\parkinsons.csv')
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, Y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = np.array(features)
    input_data_std = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_data_std)
    result = "The person has Parkinson's disease." if prediction[0] else "The person does not have Parkinson's disease."
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
