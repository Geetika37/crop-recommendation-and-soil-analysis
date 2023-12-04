from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
from PIL import Image



app = Flask(__name__, static_url_path='/static')


# Define a secret key for Flask-WTF (replace with a secure key in production)
app.config['SECRET_KEY'] = 'your_secret_key'


# Define a folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the crop recommendation model


crop_model = pickle.load(open(r'C:\Users\user\OneDrive\Desktop\crop recommendation and soil analysis\model.pkl', 'rb'))
ms=pickle.load(open(r'C:\Users\user\OneDrive\Desktop\crop recommendation and soil analysis\minmaxscaler.pkl','rb'))
sc=pickle.load(open(r'C:\Users\user\OneDrive\Desktop\crop recommendation and soil analysis\standscaler.pkl','rb'))

# Load the soil analysis model
model = tf.keras.models.load_model(r'C:\Users\user\OneDrive\Desktop\crop recommendation and soil analysis\soil_analysis3.h5')


# Define allowed extensions for image files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/soil-analysis')
def soil_analysis():
    return render_template('soil-analysis.html')


@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['rainfall'])

        # Make a crop recommendation
        predict = recommendation(N,P,K,temperature,humidity,pH,rainfall)


        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if predict[0] in crop_dict:
            crop = crop_dict[predict[0]]
            print("{} is a best crop to be cultivated ".format(crop))
        else:
            print("Sorry are not able to recommend a proper crop for this environment")
            
        return render_template('crop-recommendation.html', recommended_crop=crop)

    return render_template('crop-recommendation.html')

def recommendation(N, P, K, temperature, humidity, pH, rainfall):
    features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    transformed_features = ms.transform(features)
    transformed_features = sc.transform(transformed_features)
    prediction = crop_model.predict(transformed_features).reshape(1, -1)

    return prediction[0]






# Route for soil analysis
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if a file is included in the request
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            img = image.load_img(image_path, target_size=(220, 220, 3))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            predictions = model.predict(x)
            predicted_class = np.argmax(predictions)
            classes = ['Alluvial soil','Black Soil', 'Clay soil','Red soil',]

            predicted_class_name = classes[predicted_class]

            return render_template('soil-analysis.html', predicted_class=predicted_class_name)

    return render_template('soil-analysis.html')
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
