from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import faiss
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import mysql.connector
from werkzeug.utils import secure_filename

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

UPLOAD_FOLDER = 'uploads'
IMAGE_DB_FOLDER = 'static/image_database'
YOUR_DISTANCE_THRESHOLD = 0.5
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_new_password",  # change this
        database="face_recognition_db"
    )

def initialize_database():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_new_password"  # change this
    )
    cur = conn.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS face_recognition_db")
    conn.database = "face_recognition_db"
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_path VARCHAR(255) UNIQUE,
            embedding LONGBLOB
        )
    """)
    conn.commit()
    cur.close()
    conn.close()



def get_face_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                embedding = model(face.unsqueeze(0).to(device)).cpu().numpy()
            return embedding
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
    return None

def store_embedding(image_path, embedding):
    conn = get_mysql_connection()
    cur = conn.cursor()
    embedding_blob = pickle.dumps(embedding[0])
    cur.execute("INSERT INTO face_embeddings (image_path, embedding) VALUES (%s, %s)", (image_path, embedding_blob))
    conn.commit()
    cur.close()
    conn.close()

def embedding_exists(image_path):
    conn = get_mysql_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM face_embeddings WHERE image_path = %s", (image_path,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists

def load_all_embeddings():
    conn = get_mysql_connection()
    cur = conn.cursor()
    cur.execute("SELECT image_path, embedding FROM face_embeddings")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    paths, vectors = [], []
    for path, blob in rows:
        paths.append(path)
        vectors.append(pickle.loads(blob))
    return np.array(vectors).astype('float32'), paths



def search_similar_faces(uploaded_path, index, paths):
    embedding = get_face_embedding(uploaded_path)
    if embedding is not None:
        query = embedding[0].reshape(1, -1).astype('float32')
        k = min(5, len(paths))
        distances, indices = index.search(query, k)
        results = []
        for j, i in enumerate(indices[0]):
            if distances[0][j] < YOUR_DISTANCE_THRESHOLD:
                static_path = f"/{paths[i]}" if "static" in paths[i] else f"/static/image_database/{os.path.basename(paths[i])}"
                results.append((static_path, float(distances[0][j])))
        return results
    return []

# -------------------- Upload Route --------------------

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    embedding = get_face_embedding(filepath)
    if embedding is None:
        return jsonify({"error": "No face detected"}), 400

    if not embedding_exists(filepath):
        store_embedding(filepath, embedding)

    embeddings, paths = load_all_embeddings()
    if len(embeddings) == 0:
        return jsonify({"message": "Database is empty"})

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    results = search_similar_faces(filepath, index, paths)
    return jsonify({"matches": results}) if results else jsonify({"message": "No similar faces found"})



def initialize_image_database():
    print("📦 Initializing static/image_database folder...")

    for filename in os.listdir(IMAGE_DB_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(IMAGE_DB_FOLDER, filename)

            if not embedding_exists(full_path):
                embedding = get_face_embedding(full_path)
                if embedding is not None:
                    store_embedding(full_path, embedding)
                    print(f"✅ Embedded: {filename}")
                else:
                    print(f"❌ No face found in: {filename}")
            else:
                print(f"⏩ Already stored: {filename}")



@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')



initialize_database()
initialize_image_database()

if __name__ == '__main__':
    try:
        initialize_database()
        initialize_image_database()
        app.run(debug=True, port=5009)
    except Exception as e:
        print(f"❌ Error during startup: {e}")
