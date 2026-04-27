import os
import base64
import sqlite3
import numpy as np
import cv2
from mediapipe.python.solutions import face_detection  # ✅ FIXED IMPORT
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey_face_rec_app_007'

DB_PATH = 'database/face_recognition.db'
FACES_DIR = 'static/faces'

os.makedirs('database', exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# ✅ Mediapipe setup (fixed)
face_detection_model = face_detection.FaceDetection(min_detection_confidence=0.5)

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
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection_model.process(rgb)  # ✅ FIXED

    if not results.detections:
        return None

    if len(results.detections) > 1:
        return "multiple"

    bbox = results.detections[0].location_data.relative_bounding_box
    h, w, _ = img.shape

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    w_box = int(bbox.width * w)
    h_box = int(bbox.height * h)

    face = img[y:y+h_box, x:x+w_box]
    return face

# ---------- IMAGE COMPARISON ----------
def compare_faces(img1, img2):
    img1 = cv2.resize(img1, (200, 200))
    img2 = cv2.resize(img2, (200, 200))

    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

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

        best_score = -1
        best_user = None

        for name, path in users:
            saved = cv2.imread(path)
            score = compare_faces(face, saved)

            if score > best_score:
                best_score = score
                best_user = name

        if best_score > 0.7:
            return jsonify({'success': True, 'message': f'User: {best_user}'})
        else:
            return jsonify({'success': False, 'message': 'No match'})

    return render_template('verify.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
