import os
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet

embedder = FaceNet()
data = {'embeddings': [], 'names': []}
dataset_path = 'dataset'

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        path = os.path.join(person_folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (160, 160))
        embedding = embedder.embeddings([resized])[0]
        data['embeddings'].append(embedding)
        data['names'].append(person_name)

os.makedirs('embeddings', exist_ok=True)
with open('embeddings/embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✅ Embeddings generated and saved.")
