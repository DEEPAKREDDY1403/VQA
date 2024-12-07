from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from transformers import ViltProcessor, ViltForQuestionAnswering, BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
    def forward(self, x):
        return self.resnet(x)
class LSTMModule(nn.Module):
    def __init__(self):
        super(LSTMModule, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 256)
    def forward(self, x):
        outputs = self.bert_model(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token)
class GNNModule(nn.Module):
    def __init__(self):
        super(GNNModule, self).__init__()
        self.fc = nn.Linear(256 + 2048, 512)
    def forward(self, cnn_features, text_features):
        combined = torch.cat((cnn_features, text_features), dim=-1)
        return self.fc(combined)
logging.basicConfig(level=logging.INFO)
try:
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    logging.info("Model and processor loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if not email or not password:
            flash('Please enter your details', 'error')
            return render_template('login.html')
        user = User.query.filter_by(email=email).first()
        if user:
            if bcrypt.check_password_hash(user.password, password):
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Please, give correct login details', 'error')
                return render_template('login.html')
        else:
            flash('Please, register first', 'error')
            return render_template('login.html')
    return render_template('login.html')
@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Question text is required.'}), 400
        logging.info(f"User question: {text}")
        if 'image' not in request.files:
            return jsonify({'error': 'Image file is required.'}), 400
        image_file = request.files['image']
        image = Image.open(image_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = preprocess(image).unsqueeze(0)
        cnn_model = CNNModule()
        cnn_features = cnn_model(image_tensor)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        lstm_model = LSTMModule()
        text_features = lstm_model(inputs['input_ids'])
        gnn_model = GNNModule()
        gnn_output = gnn_model(cnn_features, text_features)
        encoding = processor(images=image, text=text, return_tensors="pt")
        if 'input_ids' not in encoding or 'pixel_values' not in encoding:
            raise ValueError("Both 'image' and 'text' inputs are required.")
        outputs = model(**encoding)
        logits = outputs.logits
        top_values, top_indices = torch.topk(logits, 3, dim=-1)
        top_predictions = []
        for idx, value in zip(top_indices[0], top_values[0]):
            answer = model.config.id2label[idx.item()]
            confidence = torch.softmax(value, dim=-1).item() * 100
            top_predictions.append({'answer': answer, 'confidence': f"{confidence:.2f}%"})
        logging.info(f"Top predictions: {top_predictions}")
        response = {
            'predicted_answer': top_predictions[0]['answer'],
            'confidence': top_predictions[0]['confidence'],
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
