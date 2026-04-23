import os
import base64
import pickle
import sqlite3
import numpy as np
import cv2
import face_recognition
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps

app = Flask(__name__)
app.secret_key = 'supersecretkey_face_rec_app_007'

DB_PATH = 'database/face_recognition.db'
FACES_DIR = 'static/faces'

# Ensure directories exist
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
            encoding BLOB NOT NULL,
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
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == 'havish' and password == 'havishfp007':
            session['admin_logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('verify'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, image_path, created_at FROM users ORDER BY created_at DESC')
    users = c.fetchall()
    conn.close()
    return render_template('dashboard.html', users=users)

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        image_data = request.form.get('image')
        
        if not name or not image_data:
            return jsonify({'success': False, 'message': 'Name and image are required.'})
        
        try:
            # Parse base64 image data from data URI
            header, encoded = image_data.split(",", 1)
            data = base64.b64decode(encoded)
            
            # Convert bytes to numpy array for OpenCV
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_img)
            
            if len(face_locations) == 0:
                return jsonify({'success': False, 'message': 'No face detected. Please ensure your face is clearly visible.'})
            elif len(face_locations) > 1:
                return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one face is in the frame.'})
            
            # Generate encoding
            encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
            
            # Save face image to disk
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).replace(' ', '_')
            image_filename = f"{safe_name}_{os.urandom(4).hex()}.jpg"
            image_path = os.path.join(FACES_DIR, image_filename)
            cv2.imwrite(image_path, img)
            
            # Save user data to DB
            encoding_bytes = pickle.dumps(encoding)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('INSERT INTO users (name, image_path, encoding) VALUES (?, ?, ?)', (name, image_path, encoding_bytes))
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': f'User "{name}" registered successfully!'})
            
        except Exception as e:
            print("Error during registration:", str(e))
            return jsonify({'success': False, 'message': 'Internal server error during registration.'})

    return render_template('register.html')

@app.route('/delete_user/<int:id>', methods=['POST'])
@login_required
def delete_user(id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT image_path FROM users WHERE id = ?', (id,))
    user = c.fetchone()
    if user:
        image_path = user[0]
        if os.path.exists(image_path):
            os.remove(image_path)
        c.execute('DELETE FROM users WHERE id = ?', (id,))
        conn.commit()
        flash('User deleted successfully.', 'success')
    conn.close()
    return redirect(url_for('dashboard'))

@app.route('/update_user/<int:id>', methods=['GET', 'POST'])
@login_required
def update_user(id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if request.method == 'POST':
        new_name = request.form.get('name')
        if new_name:
            c.execute('UPDATE users SET name = ? WHERE id = ?', (new_name, id))
            conn.commit()
            flash('User updated successfully.', 'success')
            conn.close()
            return redirect(url_for('dashboard'))
            
    c.execute('SELECT id, name FROM users WHERE id = ?', (id,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('dashboard'))
        
    return render_template('update_user.html', user=user)

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        image_data = request.form.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided.'})
            
        try:
            # Parse base64 image data
            header, encoded = image_data.split(",", 1)
            data = base64.b64decode(encoded)
            
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Find face in captured image
            face_locations = face_recognition.face_locations(rgb_img)
            
            if len(face_locations) == 0:
                return jsonify({'success': False, 'message': 'No face detected in the frame.'})
            elif len(face_locations) > 1:
                return jsonify({'success': False, 'message': 'Multiple faces detected.'})
                
            # Get the encoding for the face
            unknown_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
            
            # Fetch all stored users
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT name, encoding FROM users')
            users = c.fetchall()
            conn.close()
            
            if not users:
                return jsonify({'success': False, 'message': 'No registered users in the database.'})
                
            known_encodings = []
            known_names = []
            for user in users:
                known_names.append(user[0])
                known_encodings.append(pickle.loads(user[1]))
                
            # Compare face
            matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                matched_name = known_names[best_match_index]
                return jsonify({'success': True, 'message': f'User Identified: {matched_name}'})
            else:
                return jsonify({'success': False, 'message': 'Face Not Matched'})
                
        except Exception as e:
            print("Error during verification:", str(e))
            return jsonify({'success': False, 'message': 'Internal server error during verification.'})

    return render_template('verify.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
