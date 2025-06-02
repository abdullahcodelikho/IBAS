import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras_facenet import FaceNet
import joblib
from datetime import datetime
import csv
import os

embedder = FaceNet()
classifier = None
if os.path.exists('embeddings/svm_classifier.pkl'):
    classifier = joblib.load('embeddings/svm_classifier.pkl')
    print("✅ Classifier loaded")
else:
    print("⚠️ Classifier not found. Please train it first.")

seen = set()

face_cascade_path = r'D:\Bachelors\6th Semester\Digital Image Processing\Semester Project\IBAS_Project\haarcascade_frontalface_alt2.xml'
if not os.path.exists(face_cascade_path):
    raise FileNotFoundError(f"Haarcascade file not found: {face_cascade_path}")
face_cascade = cv2.CascadeClassifier(face_cascade_path)

CAMERA_URL = 'http://198.168.1.13:8080/shot.jpg'

class IBASLiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IBAS - Live Attendance System")
        self.root.geometry("960x740")
        self.root.resizable(False, False)

        self.style = ttk.Style()
        self.theme = 'light'
        self.set_theme(self.theme)

        topbar = ttk.Frame(self.root, padding=10)
        topbar.pack(fill='x')

        ttk.Label(topbar, text="Image-Based Attendance System", font=("Segoe UI", 16, "bold")).pack(side='left')
        ttk.Button(topbar, text="Toggle Theme", command=self.toggle_theme).pack(side='right')

        self.video_frame = ttk.Label(self.root)
        self.video_frame.pack(pady=10)

        self.status_text = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_text, font=("Segoe UI", 11)).pack(pady=5)

        control_bar = ttk.Frame(self.root, padding=10)
        control_bar.pack(fill='x', pady=10)

        ttk.Button(control_bar, text="📷 Capture Dataset", command=self.capture_dataset).pack(side='left', padx=5)
        ttk.Button(control_bar, text="🔗 Generate Embeddings", command=lambda: self.run_script("face_embedding.py")).pack(side='left', padx=5)
        ttk.Button(control_bar, text="🧠 Train Classifier", command=lambda: self.run_script("train_classifier.py")).pack(side='left', padx=5)
        ttk.Button(control_bar, text="📁 View Attendance", command=self.view_attendance).pack(side='left', padx=5)
        ttk.Button(control_bar, text="❌ Exit", command=self.quit_app).pack(side='right', padx=5)

        self.cap = cv2.VideoCapture(CAMERA_URL)
        self.frame_count = 0

        self.update_frame()

    def set_theme(self, theme):
        if theme == 'light':
            self.root.configure(bg='#f8f8f8')
            self.style.theme_use('default')
            self.style.configure('.', background='#f8f8f8', foreground='black', font=('Segoe UI', 10))
        else:
            self.root.configure(bg='#2b2b2b')
            self.style.theme_use('clam')
            self.style.configure('.', background='#2b2b2b', foreground='white', font=('Segoe UI', 10))

    def toggle_theme(self):
        self.theme = 'dark' if self.theme == 'light' else 'light'
        self.set_theme(self.theme)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            display_frame = frame.copy()

            if self.frame_count % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) > 0 and classifier is not None:
                    for (x, y, w, h) in faces:
                        padding = 10
                        x1 = max(x - padding, 0)
                        y1 = max(y - padding, 0)
                        x2 = min(x + w + padding, frame.shape[1])
                        y2 = min(y + h + padding, frame.shape[0])
                        face_img = frame[y1:y2, x1:x2]

                        try:
                            resized_face = cv2.resize(face_img, (160, 160))
                        except:
                            continue

                        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                        embedding = embedder.embeddings([rgb_face])[0]

                        preds = classifier.predict_proba([embedding])[0]
                        best_idx = np.argmax(preds)
                        name = classifier.classes_[best_idx]
                        confidence = preds[best_idx]

                        if confidence > 0.90:
                            self.status_text.set(f"Recognized: {name} ({confidence:.2f})")
                            if name not in seen:
                                seen.add(name)
                                self.log_attendance(name)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{name} ({confidence:.2f})", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            self.status_text.set("Face detected but not recognized")
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(display_frame, "Unrecognized", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    self.status_text.set("No face detected" if classifier else "Classifier not trained")

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.root.after(50, self.update_frame)

    def capture_dataset(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("Input", "Enter person's name:")
        if name:
            self.run_script('capture_dataset.py', {'PERSON_NAME': name})

    def run_script(self, script_name, env_vars=None):
        import sys
        import subprocess
        python_path = sys.executable
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        subprocess.Popen([python_path, script_name], env=env)

    def view_attendance(self):
        if not os.path.exists("attendance.csv"):
            from tkinter import messagebox
            messagebox.showinfo("Log", "No attendance file found.")
        else:
            os.system("notepad attendance.csv")

    def log_attendance(self, name):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, now])
        print(f"[LOGGED] {name} at {now}")

    def quit_app(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = IBASLiveApp(root)
    root.mainloop()
