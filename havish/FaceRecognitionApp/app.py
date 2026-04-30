import os
import base64
import sqlite3
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey_face_rec_app_007'

DB_PATH = 'database/face_recognition.db'
FACES_DIR = 'static/faces'

os.makedirs('database', exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

from deepface import DeepFace

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return redirect(url_for('verify'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == 'havish' and request.form.get('password') == 'havishfp007':
            session['admin_logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('verify'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    users = conn.execute('SELECT id, name, image_path FROM users').fetchall()
    conn.close()
    return render_template('dashboard.html', users=users)

# ---------- FACE DETECTION ----------
def detect_single_face(img):
    try:
        # Using retinaface for better accuracy, or opencv for speed
        results = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=False)
        
        if len(results) == 0 or results[0]['confidence'] < 0.9:
            return None
        
        if len(results) > 1:
            return "multiple"

        # Return the face crop as a numpy array (BGR)
        face = results[0]['face']
        # DeepFace returns face in 0-1 range sometimes depending on version, 
        # but usually it's a normalized float array or uint8. 
        # We need it in uint8 for cv2.imwrite if we use it that way.
        if face.dtype == np.float32 or face.dtype == np.float64:
            face = (face * 255).astype(np.uint8)
            
        return face
    except Exception as e:
        print(f"Detection error: {e}")
        return None

# ---------- IMAGE COMPARISON ----------
def compare_faces(img1, img2):
    try:
        # Using ArcFace model for state-of-the-art accuracy
        result = DeepFace.verify(img1, img2, model_name='ArcFace', detector_backend='opencv', enforce_detection=False)
        return result['verified'], result['distance']
    except Exception as e:
        print(f"Comparison error: {e}")
        return False, 1.0

# ---------- REGISTER ----------
@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        image_data = request.form.get('image')

        header, encoded = image_data.split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)

        face = detect_single_face(img)

        if face is None:
            return jsonify({'success': False, 'message': 'No face detected'})
        if face == "multiple":
            return jsonify({'success': False, 'message': 'Multiple faces detected'})

        filename = f"{name}_{os.urandom(4).hex()}.jpg"
        path = os.path.join(FACES_DIR, filename)
        cv2.imwrite(path, face)

        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO users (name, image_path) VALUES (?, ?)', (name, path))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'User registered'})

    return render_template('register.html')

# ---------- VERIFY ----------
@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        image_data = request.form.get('image')

        header, encoded = image_data.split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)

        face = detect_single_face(img)

        if face is None:
            return jsonify({'success': False, 'message': 'No face detected'})
        if face == "multiple":
            return jsonify({'success': False, 'message': 'Multiple faces detected'})

        conn = sqlite3.connect(DB_PATH)
        users = conn.execute('SELECT name, image_path FROM users').fetchall()
        conn.close()

        best_distance = 1.0
        best_user = None

        for name, path in users:
            saved = cv2.imread(path)
            is_match, distance = compare_faces(face, saved)

            if is_match and distance < best_distance:
                best_distance = distance
                best_user = name

        if best_user:
            return jsonify({'success': True, 'message': f'User: {best_user}'})
        else:
            return jsonify({'success': False, 'message': 'No match'})

    return render_template('verify.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
