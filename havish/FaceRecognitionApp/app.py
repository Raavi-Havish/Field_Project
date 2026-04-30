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

# Global variable for InsightFace model
face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        try:
            from insightface.app import FaceAnalysis
            # Use 'buffalo_sc' for a good balance of size and accuracy on Lambda
            # Models are stored in /tmp to comply with Lambda's read-only filesystem
            face_app = FaceAnalysis(name='buffalo_sc', root='/tmp/.insightface', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"Error initializing InsightFace: {e}")
    return face_app

# ---------- FACE DETECTION & EMBEDDING ----------
def get_face_embedding(img):
    app = get_face_app()
    if app is None: return None
    
    faces = app.get(img)
    
    if len(faces) == 0:
        return None
    if len(faces) > 1:
        return "multiple"

    # Return the embedding vector
    return faces[0].embedding

# ---------- COSINE SIMILARITY ----------
def compare_embeddings(emb1, emb2):
    if emb1 is None or emb2 is None: return 0
    # Cosine similarity formula
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return sim

# ---------- REGISTER ----------
@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        image_data = request.form.get('image')

        header, encoded = image_data.split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)

        embedding = get_face_embedding(img)

        if embedding is None:
            return jsonify({'success': False, 'message': 'No face detected'})
        if embedding == "multiple":
            return jsonify({'success': False, 'message': 'Multiple faces detected'})

        # Save the original image (or we could save the embedding to DB for speed)
        filename = f"{name}_{os.urandom(4).hex()}.jpg"
        path = os.path.join(FACES_DIR, filename)
        cv2.imwrite(path, img)

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

        embedding = get_face_embedding(img)

        if embedding is None:
            return jsonify({'success': False, 'message': 'No face detected'})
        if embedding == "multiple":
            return jsonify({'success': False, 'message': 'Multiple faces detected'})

        conn = sqlite3.connect(DB_PATH)
        users = conn.execute('SELECT name, image_path FROM users').fetchall()
        conn.close()

        best_score = 0
        best_user = None

        for name, path in users:
            saved_img = cv2.imread(path)
            saved_embedding = get_face_embedding(saved_img)
            
            if isinstance(saved_embedding, np.ndarray):
                score = compare_embeddings(embedding, saved_embedding)
                if score > best_score:
                    best_score = score
                    best_user = name

        # Threshold for ArcFace (buffalo_sc) cosine similarity is usually around 0.4-0.5
        if best_score > 0.45:
            return jsonify({'success': True, 'message': f'User: {best_user}'})
        else:
            return jsonify({'success': False, 'message': 'No match'})

    return render_template('verify.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
