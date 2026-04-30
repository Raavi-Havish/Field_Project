import os
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

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
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
    # Added created_at to avoid IndexError in template
    users = conn.execute('SELECT id, name, image_path, created_at FROM users').fetchall()
    conn.close()
    return render_template('dashboard.html', users=users)

@app.route('/delete_user/<int:id>', methods=['POST'])
@login_required
def delete_user(id):
    conn = sqlite3.connect(DB_PATH)
    user = conn.execute('SELECT image_path FROM users WHERE id = ?', (id,)).fetchone()
    if user:
        # We don't delete from /tmp as it's ephemeral, but we could
        conn.execute('DELETE FROM users WHERE id = ?', (id,))
        conn.commit()
    conn.close()
    flash('User deleted successfully', 'success')
    return redirect(url_for('dashboard'))

@app.route('/update_user/<int:id>', methods=['GET', 'POST'])
@login_required
def update_user(id):
    # For now, just a placeholder or basic edit
    if request.method == 'POST':
        name = request.form.get('name')
        conn = sqlite3.connect(DB_PATH)
        conn.execute('UPDATE users SET name = ? WHERE id = ?', (name, id))
        conn.commit()
        conn.close()
        flash('User updated', 'success')
        return redirect(url_for('dashboard'))
    
    conn = sqlite3.connect(DB_PATH)
    user = conn.execute('SELECT id, name FROM users WHERE id = ?', (id,)).fetchone()
    conn.close()
    return render_template('register.html', user=user) # Reuse register for edit

# Global variable for Mediapipe model
face_embedder = None
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_embedder/face_embedder/float16/1/face_embedder.task"
MODEL_PATH = "/tmp/face_embedder.task"

def get_face_embedder():
    global face_embedder
    if face_embedder is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import requests

            # Download model if not exists
            if not os.path.exists(MODEL_PATH):
                print("Downloading Mediapipe Face Embedder model...")
                r = requests.get(MODEL_URL, allow_redirects=True)
                with open(MODEL_PATH, 'wb') as f:
                    f.write(r.content)
            
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.FaceEmbedderOptions(base_options=base_options)
            face_embedder = vision.FaceEmbedder.create_from_options(options)
        except Exception as e:
            print(f"Error initializing Mediapipe: {e}")
    return face_embedder

# ---------- FACE DETECTION & EMBEDDING ----------
def get_face_embedding(img):
    if img is None: return None
    embedder = get_face_embedder()
    if embedder is None: return None
    
    import mediapipe as mp
    # Convert OpenCV BGR to RGB
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = embedder.embed(mp_image)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None
    
    if not result.embeddings:
        return None
    
    # If multiple faces are detected, result.embeddings will have multiple entries
    if len(result.embeddings) > 1:
        return "multiple"

    return result.embeddings[0].float_vector

# ---------- COSINE SIMILARITY ----------
def compare_embeddings(emb1, emb2):
    if emb1 is None or emb2 is None: return 0
    # Use numpy for cosine similarity
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
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
            if saved_img is None: continue
            
            saved_embedding = get_face_embedding(saved_img)
            
            # Mediapipe returns a list (float_vector)
            if saved_embedding and not isinstance(saved_embedding, str):
                score = compare_embeddings(embedding, saved_embedding)
                if score > best_score:
                    best_score = score
                    best_user = name

        # Threshold for Mediapipe Face Embedder cosine similarity is usually around 0.8
        if best_score > 0.8:
            return jsonify({'success': True, 'message': f'User: {best_user}'})
        else:
            return jsonify({'success': False, 'message': 'No match'})

    return render_template('verify.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
