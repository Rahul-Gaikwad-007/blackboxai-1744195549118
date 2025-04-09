from flask import Flask, jsonify, request, render_template
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

# Load and train the model
def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    joblib.dump(model, 'iris_model.pkl')

train_model()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/iris', methods=['GET'])
def get_iris_data():
    # Load the Iris dataset
    iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
                             header=None, 
                             names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    return jsonify(iris_data.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model = joblib.load('iris_model.pkl')
    prediction = model.predict([data['features']])
    
    # Map the prediction index to the class name
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    predicted_class = class_names[int(prediction[0])]
    
    return jsonify({'prediction': predicted_class})  # Return the class name

if __name__ == '__main__':
    app.run(debug=True, port=8000)
