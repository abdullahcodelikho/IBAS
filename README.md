# IBAS - Image-Based Attendance System

A smart attendance system that uses face recognition to automatically mark attendance. The system includes features to prevent photo/video spoofing and provides a user-friendly interface for managing students and attendance records.

## Features

- **Face Recognition Attendance**
  - Real-time face detection and recognition
  - Automatic attendance marking
  - Support for multiple classes

- **Student Management**
  - Add new students with photos
  - Edit student information
  - Delete student records
  - Organize students by classes

- **Anti-Spoofing Measures**
  - Liveness detection using eye blink detection
  - Head movement verification
  - Prevents photo and video spoofing

- **Dynamic Camera Support**
  - Change camera URL at runtime
  - Support for IP cameras and webcams
  - Easy switching between different video sources

- **Search Functionality**
  - Search students by roll number
  - Real-time filtering
  - Highlight matching results

## Requirements

- Python 3.7+
- OpenCV
- dlib
- face_recognition
- ttkbootstrap
- scipy
- numpy
- pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abdullahcodelikho/IBAS.git
cd IBAS
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the facial landmark predictor:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Usage

1. Run the main application:
```bash
python main_gui.py
```

2. Create a new class or select an existing one
3. Add students using the Capture tab
4. Start attendance monitoring
5. View and manage attendance records

## Project Structure

- `main_gui.py` - Main application interface
- `capture_dataset.py` - Student image capture
- `face_embedding.py` - Face encoding generation
- `train_classifier.py` - SVM classifier training
- `recognize_and_log.py` - Face recognition and attendance logging

## Contributing

Feel free to submit issues and enhancement requests! 