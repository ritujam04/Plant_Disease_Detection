import string
#from flask import Flask, redirect, render_template, url_for, request
from markupsafe import Markup
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.disease import disease_dic

import os
#import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_pymongo import PyMongo




# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------


# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

# disease prediction
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()



def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction



app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Set to 32MB (or adjust as needed)
app.secret_key = "4b8f0821a6f74a3bb7d6c3f7e7c7a1d22871e0b529ebda8e66fa24623e8a3b5f"

# MongoDB Atlas Connection
app.config["MONGO_URI"] = "mongodb+srv://rituja0412:FjUH75668F6XouHM@cluster0.r2uid.mongodb.net/flask_app_db"
mongo = PyMongo(app)


@app.route("/")
def hello_world():
    return render_template("index.html")
    

# ------------------- Serve Pages -------------------

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/upload")
def upload_page():
    if "user_id" not in session:
        return redirect(url_for("login_page"))
    return render_template("upload.html")

# ------------------- Register User -------------------
@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if not name or not email or not password:
        return "All fields are required", 400

    existing_user = mongo.db.users.find_one({"email": email})
    if existing_user:
        return "Email already exists", 400

    hashed_password = generate_password_hash(password)
    mongo.db.users.insert_one({"name": name, "email": email, "password": hashed_password})

    return redirect(url_for("login_page"))

# ------------------- Login User -------------------
@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    if not email or not password:
        return "Email and password are required", 400

    user = mongo.db.users.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        return "Invalid email or password", 401

    session["user_id"] = str(user["_id"])  # Store user in session

    return redirect(url_for("upload_page"))


# @app.route('/disease-predict', methods=['GET', 'POST'])
# @login_required
# def disease_prediction():
#     title = '- Disease Detection'
#     return render_template('disease.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = '- Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

# ------------------- Logout -------------------
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login_page"))


if __name__ == "__main__":
    app.run(debug=True,port=8000)
