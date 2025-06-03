import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime
import urllib.request
import csv
import dlib
from scipy.spatial import distance as dist
import bz2
import requests
from tqdm import tqdm

BASE_DATASET_PATH = r'D:\Bachelors\6th Semester\Digital Image Processing\Semester Project\IBAS_Project\dataset'
ATTENDANCE_FOLDER = r'D:\Bachelors\6th Semester\Digital Image Processing\Semester Project\IBAS_Project\attendance_logs'
CAMERA_URL = 'http://192.168.1.13:8080/shot.jpg'
CONFIDENCE_THRESHOLD = 0.60
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
PREDICTOR_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

def get_attendance_filename(current_class):
    today = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(ATTENDANCE_FOLDER, f"attendance_{current_class}_{today}.csv")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, boxes)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

def download_predictor():
    if os.path.exists(PREDICTOR_PATH):
        return True

    try:
        # Download the compressed file
        response = requests.get(PREDICTOR_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Save the compressed file
        compressed_path = PREDICTOR_PATH + '.bz2'
        with open(compressed_path, 'wb') as f, tqdm(
            desc="Downloading facial landmark predictor",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

        # Decompress the file
        with bz2.open(compressed_path, 'rb') as source, open(PREDICTOR_PATH, 'wb') as dest:
            dest.write(source.read())

        # Remove the compressed file
        os.remove(compressed_path)
        return True
    except Exception as e:
        print(f"Error downloading predictor: {str(e)}")
        return False

class IBASApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IBAS - Smart Attendance System")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 700)
        self.root.state('zoomed')

        # Download predictor if not exists
        if not download_predictor():
            messagebox.showerror("Error", "Failed to download facial landmark predictor. Liveness detection will be disabled.")
            self.predictor = None
            self.face_detector = None
        else:
            # Initialize face detector and facial landmark predictor
            self.face_detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

        # Liveness detection variables
        self.blink_counter = 0
        self.head_movement_counter = 0
        self.liveness_check_active = False
        self.eye_aspect_ratio_threshold = 0.25
        self.head_movement_threshold = 0.1
        self.required_blinks = 2
        self.required_head_movements = 1
        self.previous_head_position = None

        self.current_class = None
        self.images = []
        self.classNames = []
        self.rollNumbers = []
        self.encodeListKnown = []
        self.seen = set()
        self.attendance_running = False
        self.attendance_file = None
        self.current_camera_url = CAMERA_URL  # Default camera URL

        self.build_ui()

        self.root.bind('<s>', lambda e: self.start_video_feed())
        self.root.bind('<q>', lambda e: self.stop_attendance())

    def calculate_eye_aspect_ratio(self, eye_landmarks):
        # Calculate the vertical distances
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Calculate the horizontal distance
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        # Calculate the eye aspect ratio
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def get_eye_landmarks(self, shape):
        # Get the landmarks for both eyes
        left_eye = np.array([(shape.part(36+i).x, shape.part(36+i).y) for i in range(6)])
        right_eye = np.array([(shape.part(42+i).x, shape.part(42+i).y) for i in range(6)])
        return left_eye, right_eye

    def detect_blink(self, frame):
        if self.predictor is None:
            return False, frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        for face in faces:
            shape = self.predictor(gray, face)
            left_eye, right_eye = self.get_eye_landmarks(shape)
            
            # Calculate eye aspect ratio for both eyes
            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Draw eye landmarks
            for eye in [left_eye, right_eye]:
                for (x, y) in eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Check for blink
            if ear < self.eye_aspect_ratio_threshold:
                self.blink_counter += 1
                return True, frame
            return False, frame
        return False, frame

    def detect_head_movement(self, frame):
        if self.predictor is None:
            return False, frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        for face in faces:
            shape = self.predictor(gray, face)
            nose_tip = np.array([shape.part(30).x, shape.part(30).y])
            
            if self.previous_head_position is not None:
                movement = np.linalg.norm(nose_tip - self.previous_head_position)
                if movement > self.head_movement_threshold:
                    self.head_movement_counter += 1
                    self.previous_head_position = nose_tip
                    return True, frame
            
            self.previous_head_position = nose_tip
            return False, frame
        return False, frame

    def build_ui(self):
        notebook = tb.Notebook(self.root)
        notebook.pack(fill='both', expand=True)

        self.frame_attendance = tb.Frame(notebook)
        self.frame_capture = tb.Frame(notebook)
        self.frame_manage = tb.Frame(notebook)

        notebook.add(self.frame_attendance, text='📸 Attendance')
        notebook.add(self.frame_capture, text='🧍 Capture')
        notebook.add(self.frame_manage, text=' Manage Students')

        # Attendance tab UI
        top_frame = tb.Frame(self.frame_attendance)
        top_frame.grid(row=0, column=0, sticky="ew", pady=10, padx=10)

        tb.Label(top_frame, text="Select Class:").grid(row=0, column=0, padx=5)
        self.class_combobox = tb.Combobox(top_frame, state="readonly")
        self.class_combobox.grid(row=0, column=1, padx=5)
        tb.Button(top_frame, text="Load Class", command=self.load_class).grid(row=0, column=2, padx=5)
        tb.Button(top_frame, text="New Class", command=self.create_new_class).grid(row=0, column=3, padx=5)
        tb.Button(top_frame, text="Start Attendance (S)", command=self.start_video_feed).grid(row=0, column=4, padx=5)
        tb.Button(top_frame, text="Stop Attendance (Q)", command=self.stop_attendance).grid(row=0, column=5, padx=5)

        self.status_label = tb.Label(self.frame_attendance, text="Status: Idle", bootstyle=INFO)
        self.status_label.grid(row=1, column=0, sticky="w", padx=10)

        self.video_label = tb.Label(self.frame_attendance)
        self.video_label.grid(row=2, column=0, pady=10, padx=10)

        self.tree = tb.Treeview(self.frame_attendance, columns=("Roll No", "Name", "Time"), show='headings', height=8)
        self.tree.heading("Roll No", text="Roll No")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Time", text="Timestamp")
        self.tree.grid(row=3, column=0, padx=10, sticky='nsew')

        # Add search frame
        search_frame = tb.Frame(self.frame_attendance)
        search_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        
        tb.Label(search_frame, text="Search Roll No:").pack(side='left', padx=5)
        self.search_entry = tb.Entry(search_frame)
        self.search_entry.pack(side='left', padx=5)
        tb.Button(search_frame, text="Search", command=self.search_roll).pack(side='left', padx=5)
        tb.Button(search_frame, text="Clear", command=self.clear_search).pack(side='left', padx=5)

        # Capture tab UI
        capture_form = tb.Frame(self.frame_capture)
        capture_form.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        tb.Label(capture_form, text="Roll No:").grid(row=0, column=0, padx=5)
        self.roll_entry = tb.Entry(capture_form)
        self.roll_entry.grid(row=0, column=1, padx=5)

        tb.Label(capture_form, text="Name:").grid(row=0, column=2, padx=5)
        self.name_entry = tb.Entry(capture_form)
        self.name_entry.grid(row=0, column=3, padx=5)

        tb.Label(capture_form, text="Class:").grid(row=0, column=4, padx=5)
        self.capture_class_combobox = tb.Combobox(capture_form, state="readonly")
        self.capture_class_combobox.grid(row=0, column=5, padx=5)

        tb.Button(capture_form, text="Capture", command=self.capture_image).grid(row=0, column=6, padx=5)

        self.capture_preview = tb.Label(self.frame_capture)
        self.capture_preview.grid(row=1, column=0, padx=10, pady=10)

        # Manage Students tab UI
        manage_top_frame = tb.Frame(self.frame_manage)
        manage_top_frame.grid(row=0, column=0, sticky="ew", pady=10, padx=10)

        tb.Label(manage_top_frame, text="Select Class:").grid(row=0, column=0, padx=5)
        self.manage_class_combobox = tb.Combobox(manage_top_frame, state="readonly")
        self.manage_class_combobox.grid(row=0, column=1, padx=5)
        tb.Button(manage_top_frame, text="Load Students", command=self.load_students).grid(row=0, column=2, padx=5)

        # Student list with edit/delete buttons
        self.student_tree = tb.Treeview(self.frame_manage, columns=("Roll No", "Name"), show='headings', height=15)
        self.student_tree.heading("Roll No", text="Roll No")
        self.student_tree.heading("Name", text="Name")
        self.student_tree.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        # Buttons frame
        button_frame = tb.Frame(self.frame_manage)
        button_frame.grid(row=2, column=0, pady=10)
        tb.Button(button_frame, text="Edit Selected", command=self.edit_student).grid(row=0, column=0, padx=5)
        tb.Button(button_frame, text="Delete Selected", command=self.delete_student).grid(row=0, column=1, padx=5)

        # Configure grid weights
        self.frame_manage.grid_rowconfigure(1, weight=1)
        self.frame_manage.grid_columnconfigure(0, weight=1)

        self.refresh_classes()

    def refresh_classes(self):
        classes = [d for d in os.listdir(BASE_DATASET_PATH) if os.path.isdir(os.path.join(BASE_DATASET_PATH, d))]
        self.class_combobox['values'] = classes
        self.capture_class_combobox['values'] = classes
        self.manage_class_combobox['values'] = classes
        if classes:
            self.class_combobox.current(0)
            self.capture_class_combobox.current(0)
            self.manage_class_combobox.current(0)

    def create_new_class(self):
        new_class = simpledialog.askstring("New Class", "Enter new class name:")
        if new_class:
            os.makedirs(os.path.join(BASE_DATASET_PATH, new_class), exist_ok=True)
            self.refresh_classes()

    def load_class(self):
        selected_class = self.class_combobox.get()
        if not selected_class:
            messagebox.showwarning("Warning", "Select a class first.")
            return
        self.images = []
        self.classNames = []
        self.rollNumbers = []
        self.seen = set()
        path = os.path.join(BASE_DATASET_PATH, selected_class)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                roll, name = os.path.splitext(img_name)[0].split("_", 1)
                self.images.append(img)
                self.rollNumbers.append(roll)
                self.classNames.append(name)
        self.encodeListKnown = findEncodings(self.images)
        self.current_class = selected_class
        self.attendance_file = get_attendance_filename(selected_class)
        if not os.path.isfile(self.attendance_file):
            df = pd.DataFrame(columns=['Roll No', 'Name', 'Time'])
            df.to_csv(self.attendance_file, index=False)
        self.status_label.config(text=f"Status: {selected_class} loaded")

    def start_video_feed(self):
        if not self.attendance_running:
            # Create URL input dialog
            url_dialog = tb.Toplevel(self.root)
            url_dialog.title("Camera URL")
            url_dialog.geometry("400x150")
            url_dialog.transient(self.root)
            url_dialog.grab_set()

            # Center the dialog
            url_dialog.update_idletasks()
            width = url_dialog.winfo_width()
            height = url_dialog.winfo_height()
            x = (url_dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (url_dialog.winfo_screenheight() // 2) - (height // 2)
            url_dialog.geometry(f'{width}x{height}+{x}+{y}')

            # Add URL input
            tb.Label(url_dialog, text="Enter Camera URL:").pack(pady=10)
            url_entry = tb.Entry(url_dialog, width=50)
            url_entry.insert(0, self.current_camera_url)  # Show current URL
            url_entry.pack(pady=5)

            def start_with_url():
                new_url = url_entry.get().strip()
                if new_url:
                    self.current_camera_url = new_url
                    self.attendance_running = True
                    self.status_label.config(text="Status: Running (press Q to stop)")
                    self.update_video()
                    url_dialog.destroy()
                else:
                    messagebox.showerror("Error", "Please enter a valid URL")

            # Add buttons
            button_frame = tb.Frame(url_dialog)
            button_frame.pack(pady=10)
            tb.Button(button_frame, text="Start", command=start_with_url).pack(side='left', padx=5)
            tb.Button(button_frame, text="Cancel", command=url_dialog.destroy).pack(side='left', padx=5)

            # Bind Enter key to start
            url_entry.bind('<Return>', lambda e: start_with_url())
            url_entry.focus_set()

    def stop_attendance(self):
        self.attendance_running = False
        self.status_label.config(text="Status: Stopped")
        self.video_label.configure(image='')

    def update_video(self):
        if not self.attendance_running:
            return
        try:
            img_resp = urllib.request.urlopen(self.current_camera_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgnp, -1)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(img_rgb)
            encodings = face_recognition.face_encodings(img_rgb, faces)

            for encoding, faceLoc in zip(encodings, faces):
                faceDis = face_recognition.face_distance(self.encodeListKnown, encoding)
                matches = face_recognition.compare_faces(self.encodeListKnown, encoding, tolerance=CONFIDENCE_THRESHOLD)
                if any(matches):
                    matchIndex = np.argmin(faceDis)
                    name = self.classNames[matchIndex].upper()
                    roll = self.rollNumbers[matchIndex]
                    if roll not in self.seen:
                        self.seen.add(roll)
                        now = datetime.now().strftime('%H:%M:%S')
                        with open(self.attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([roll, name, now])
                        self.tree.insert('', 'end', values=(roll, name, now))
                    y1, x2, y2, x1 = faceLoc
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{roll} - {name}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            img_resized = cv2.resize(img, (800, 600))
            img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {str(e)}")
            self.attendance_running = False
            messagebox.showerror("Error", f"Failed to connect to camera: {str(e)}")
            return

        self.root.after(10, self.update_video)

    def capture_image(self):
        roll = self.roll_entry.get().strip()
        name = self.name_entry.get().strip()
        selected_class = self.capture_class_combobox.get()
        if not roll or not name or not selected_class:
            messagebox.showerror("Error", "Roll No, Name, and Class are required.")
            return

        path = os.path.join(BASE_DATASET_PATH, selected_class)
        filename = os.path.join(path, f"{roll}_{name}.jpg")
        if os.path.exists(filename):
            messagebox.showerror("Error", "A student with this Roll No already exists.")
            return

        # Create URL input dialog
        url_dialog = tb.Toplevel(self.root)
        url_dialog.title("Camera URL")
        url_dialog.geometry("400x150")
        url_dialog.transient(self.root)
        url_dialog.grab_set()

        # Center the dialog
        url_dialog.update_idletasks()
        width = url_dialog.winfo_width()
        height = url_dialog.winfo_height()
        x = (url_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (url_dialog.winfo_screenheight() // 2) - (height // 2)
        url_dialog.geometry(f'{width}x{height}+{x}+{y}')

        # Add URL input
        tb.Label(url_dialog, text="Enter Camera URL:").pack(pady=10)
        url_entry = tb.Entry(url_dialog, width=50)
        url_entry.insert(0, self.current_camera_url)  # Show current URL
        url_entry.pack(pady=5)

        def start_capture():
            new_url = url_entry.get().strip()
            if new_url:
                self.current_camera_url = new_url
                url_dialog.destroy()
                
                # Check if liveness detection is available
                if self.predictor is None:
                    # If liveness detection is not available, capture image directly
                    try:
                        os.makedirs(path, exist_ok=True)
                        img_resp = urllib.request.urlopen(self.current_camera_url)
                        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                        img = cv2.imdecode(imgnp, -1)
                        cv2.imwrite(filename, img)
                        self.capture_preview.configure(image=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
                        messagebox.showinfo("Success", f"Captured image for {name} ({roll}).")
                        self.refresh_classes()
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to capture image: {str(e)}")
                    return

                # Start liveness detection
                self.liveness_check_active = True
                self.blink_counter = 0
                self.head_movement_counter = 0
                self.previous_head_position = None

                messagebox.showinfo("Liveness Check", "Please blink twice and move your head slightly to verify you are a real person.")
                
                def check_liveness():
                    if not self.liveness_check_active:
                        return

                    try:
                        img_resp = urllib.request.urlopen(self.current_camera_url)
                        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                        frame = cv2.imdecode(imgnp, -1)

                        # Perform liveness checks
                        blinked, frame = self.detect_blink(frame)
                        moved, frame = self.detect_head_movement(frame)

                        # Display status
                        status_text = f"Blinks: {self.blink_counter}/{self.required_blinks} | Head Movements: {self.head_movement_counter}/{self.required_head_movements}"
                        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Show the frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.capture_preview.imgtk = imgtk
                        self.capture_preview.configure(image=imgtk)

                        # Check if liveness requirements are met
                        if self.blink_counter >= self.required_blinks and self.head_movement_counter >= self.required_head_movements:
                            self.liveness_check_active = False
                            try:
                                os.makedirs(path, exist_ok=True)
                                cv2.imwrite(filename, frame)
                                messagebox.showinfo("Success", f"Captured image for {name} ({roll}).")
                                self.refresh_classes()
                            except Exception as e:
                                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
                            return

                        self.root.after(50, check_liveness)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to capture image: {str(e)}")
                        self.liveness_check_active = False

                check_liveness()
            else:
                messagebox.showerror("Error", "Please enter a valid URL")

        # Add buttons
        button_frame = tb.Frame(url_dialog)
        button_frame.pack(pady=10)
        tb.Button(button_frame, text="Start Capture", command=start_capture).pack(side='left', padx=5)
        tb.Button(button_frame, text="Cancel", command=url_dialog.destroy).pack(side='left', padx=5)

        # Bind Enter key to start
        url_entry.bind('<Return>', lambda e: start_capture())
        url_entry.focus_set()

    def load_students(self):
        selected_class = self.manage_class_combobox.get()
        if not selected_class:
            messagebox.showwarning("Warning", "Select a class first.")
            return

        # Clear existing items
        for item in self.student_tree.get_children():
            self.student_tree.delete(item)

        # Load students from the class directory
        path = os.path.join(BASE_DATASET_PATH, selected_class)
        for img_name in os.listdir(path):
            if img_name.endswith('.jpg'):
                roll, name = os.path.splitext(img_name)[0].split("_", 1)
                self.student_tree.insert('', 'end', values=(roll, name))

    def edit_student(self):
        selected_item = self.student_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Select a student to edit.")
            return

        roll, name = self.student_tree.item(selected_item[0])['values']
        selected_class = self.manage_class_combobox.get()

        # Create edit dialog
        edit_window = tb.Toplevel(self.root)
        edit_window.title("Edit Student")
        edit_window.geometry("300x150")
        edit_window.transient(self.root)
        edit_window.grab_set()

        tb.Label(edit_window, text="Roll No:").grid(row=0, column=0, padx=5, pady=5)
        roll_entry = tb.Entry(edit_window)
        roll_entry.insert(0, roll)
        roll_entry.grid(row=0, column=1, padx=5, pady=5)

        tb.Label(edit_window, text="Name:").grid(row=1, column=0, padx=5, pady=5)
        name_entry = tb.Entry(edit_window)
        name_entry.insert(0, name)
        name_entry.grid(row=1, column=1, padx=5, pady=5)

        def save_changes():
            new_roll = roll_entry.get().strip()
            new_name = name_entry.get().strip()
            
            if not new_roll or not new_name:
                messagebox.showerror("Error", "Roll No and Name are required.")
                return

            old_path = os.path.join(BASE_DATASET_PATH, selected_class, f"{roll}_{name}.jpg")
            new_path = os.path.join(BASE_DATASET_PATH, selected_class, f"{new_roll}_{new_name}.jpg")

            if old_path != new_path and os.path.exists(new_path):
                messagebox.showerror("Error", "A student with this Roll No already exists.")
                return

            try:
                os.rename(old_path, new_path)
                self.load_students()  # Refresh the list
                edit_window.destroy()
                messagebox.showinfo("Success", "Student information updated successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update student information: {str(e)}")

        tb.Button(edit_window, text="Save", command=save_changes).grid(row=2, column=0, columnspan=2, pady=10)

    def delete_student(self):
        selected_item = self.student_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Select a student to delete.")
            return

        roll, name = self.student_tree.item(selected_item[0])['values']
        selected_class = self.manage_class_combobox.get()

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {name} ({roll})?"):
            try:
                file_path = os.path.join(BASE_DATASET_PATH, selected_class, f"{roll}_{name}.jpg")
                os.remove(file_path)
                self.load_students()  # Refresh the list
                messagebox.showinfo("Success", "Student deleted successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete student: {str(e)}")

    def search_roll(self):
        search_text = self.search_entry.get().strip().lower()
        if not search_text:
            self.clear_search()
            return

        # Clear previous highlights
        for item in self.tree.get_children():
            self.tree.item(item, tags=())

        # Search and highlight matching items
        found = False
        for item in self.tree.get_children():
            roll_no = self.tree.item(item)['values'][0].lower()
            if search_text in roll_no:
                self.tree.selection_set(item)
                self.tree.see(item)  # Scroll to the item
                self.tree.item(item, tags=('found',))
                found = True

        if not found:
            messagebox.showinfo("Search", "No matching roll numbers found.")

        # Configure tag for highlighting
        self.tree.tag_configure('found', background='#e6f3ff')

    def clear_search(self):
        self.search_entry.delete(0, 'end')
        for item in self.tree.get_children():
            self.tree.item(item, tags=())
        self.tree.selection_remove(*self.tree.selection())

if __name__ == "__main__":
    root = tb.Window(themename="flatly")
    app = IBASApp(root)
    root.mainloop()
