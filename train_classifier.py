import pickle
from sklearn.svm import SVC
import joblib

with open('embeddings/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

model = SVC(kernel='linear', probability=True)
model.fit(data['embeddings'], data['names'])

joblib.dump(model, 'embeddings/svm_classifier.pkl')
print("✅ Classifier trained and saved.")
