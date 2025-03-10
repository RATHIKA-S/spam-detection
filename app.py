from flask import Flask, render_template, request, jsonify
import pickle

# Load the model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Preprocess the message and predict
    message_vect = vectorizer.transform([message])
    prediction = model.predict(message_vect)[0]
    label = 'Spam' if prediction == 1 else 'Ham'

    return render_template('index.html', message=message, label=label)

if __name__ == '__main__':
    app.run(debug=True)
