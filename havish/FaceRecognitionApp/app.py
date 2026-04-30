import os
import json
import base64
import sqlite3
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey_face_rec_app_007'

DB_PATH = '/tmp/face_recognition.db'
FACES_DIR = 'static/faces'

os.makedirs('/tmp', exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    try:
        c.execute('SELECT embedding FROM users LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE users ADD COLUMN embedding TEXT')
    conn.commit()
    conn.close()

init_db()

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            flash('Please log in to access the admin panel.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ─────────────────────────────────────────────
# FACE RECOGNITION (uniface — ONNX, no dlib)
# ─────────────────────────────────────────────
_detector = None
_recognizer = None
_model_error = None

def get_models():
    global _detector, _recognizer, _model_error
    if _model_error:  # Don't retry if we know it failed
        return None, None
    if _detector is None or _recognizer is None:
        try:
            from uniface.detection import RetinaFace
            from uniface.recognition import ArcFace
            _detector = RetinaFace()
            _recognizer = ArcFace()
            print("UniFace models loaded successfully.")
        except Exception as e:
            _model_error = str(e)
            print(f"CRITICAL: Failed to load UniFace models: {e}")
            return None, None
    return _detector, _recognizer

def get_face_embedding(img):
    """Detect a single face and return its ArcFace embedding."""
    if img is None:
        return None, "Image is empty"

    detector, recognizer = get_models()
    if detector is None or recognizer is None:
        err = _model_error or "Unknown error loading models"
        return None, f"Face recognition system error: {err}"

    try:
        # detect() returns (boxes, kpss)
        boxes, kpss = detector.detect(img)
    except Exception as e:
        return None, f"Detection error: {e}"

    if boxes is None or len(boxes) == 0:
        return None, "No face detected. Ensure good lighting and face the camera directly."
    if len(boxes) > 1:
        return None, "Multiple faces detected. Please ensure only one face is visible."

    try:
        # extract() takes the full image and the first detected face
        embedding = recognizer.extract(img, kpss[0])
        return embedding, None
    except Exception as e:
        return None, f"Recognition error: {e}"

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ─────────────────────────────────────────────
# ROUTES — PUBLIC
# ─────────────────────────────────────────────
@app.route('/health')
def health():
    """Diagnostic endpoint — check if models are loaded."""
    detector, recognizer = get_models()
    if detector and recognizer:
        return jsonify({'status': 'ok', 'models': 'loaded'})
    return jsonify({'status': 'error', 'detail': _model_error or 'Models not loaded'}), 500

@app.route('/')
def index():
    return redirect(url_for('verify'))

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        image_data = request.form.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image received.'})

        header, encoded = image_data.split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)

        embedding, error = get_face_embedding(img)
        if error:
            return jsonify({'success': False, 'message': error})

        conn = sqlite3.connect(DB_PATH)
        users = conn.execute('SELECT name, embedding FROM users').fetchall()
        conn.close()

        if not users:
            return jsonify({'success': False, 'message': 'No users registered yet.'})

        best_score = 0
        best_user = None

        for name, saved_embedding_str in users:
            if not saved_embedding_str:
                continue
            try:
                saved_emb = np.array(json.loads(saved_embedding_str))
                score = cosine_similarity(embedding, saved_emb)
                if score > best_score:
                    best_score = score
                    best_user = name
            except Exception:
                continue

        # ArcFace cosine similarity threshold ~0.28 is typical
        if best_score > 0.28:
            return jsonify({'success': True, 'message': f'Welcome, {best_user}! ({int(best_score * 100)}% match)'})
        else:
            return jsonify({'success': False, 'message': 'Identity not recognized. Please try again.'})

    return render_template('verify.html')

# ─────────────────────────────────────────────
# ROUTES — ADMIN ONLY
# ─────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == 'havish' and request.form.get('password') == 'havishfp007':
            session['admin_logged_in'] = True
            return redirect(url_for('dashboard'))
        flash('Invalid credentials. Please try again.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('verify'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    users = conn.execute('SELECT id, name, image_path, created_at FROM users').fetchall()
    conn.close()
    return render_template('dashboard.html', users=users)

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        image_data = request.form.get('image')

        if not name:
            return jsonify({'success': False, 'message': 'Please enter a name.'})
        if not image_data:
            return jsonify({'success': False, 'message': 'No image received.'})

        header, encoded = image_data.split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)

        embedding, error = get_face_embedding(img)
        if error:
            return jsonify({'success': False, 'message': error})

        filename = f"{name.replace(' ', '_')}_{os.urandom(4).hex()}.jpg"
        path = os.path.join(FACES_DIR, filename)
        cv2.imwrite(path, img)

        embedding_str = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding))

        conn = sqlite3.connect(DB_PATH)
        conn.execute('INSERT INTO users (name, image_path, embedding) VALUES (?, ?, ?)', (name, path, embedding_str))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': f'{name} registered successfully!'})

    return render_template('register.html')

@app.route('/delete_user/<int:id>', methods=['POST'])
@login_required
def delete_user(id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM users WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('User deleted successfully.', 'success')
    return redirect(url_for('dashboard'))

@app.route('/update_user/<int:id>', methods=['GET', 'POST'])
@login_required
def update_user(id):
    conn = sqlite3.connect(DB_PATH)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        if name:
            conn.execute('UPDATE users SET name = ? WHERE id = ?', (name, id))
            conn.commit()
            conn.close()
            flash('User name updated.', 'success')
            return redirect(url_for('dashboard'))
    user = conn.execute('SELECT id, name FROM users WHERE id = ?', (id,)).fetchone()
    conn.close()
    return render_template('update_user.html', user=user)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
