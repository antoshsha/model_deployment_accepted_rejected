from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model_accepted_rejected.pkl")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        data_array = np.array(data).reshape(1, -1)
        prediction = model.predict(data_array)
        return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
