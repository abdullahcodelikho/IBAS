import os
import cv2

name = os.getenv("PERSON_NAME") or input("Enter person's name: ")
save_path = f'dataset/{name}'
os.makedirs(save_path, exist_ok=True)

url = 'http://198.168.1.13:8080/shot.jpg'
cap = cv2.VideoCapture(url)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    cv2.imshow("Capture Dataset - Press SPACE to Save, ESC to Exit", frame)
    key = cv2.waitKey(1)
    if key == 32:
        img_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"[SAVED] {img_name}")
        count += 1
    elif key == 27 or count >= 20:
        print("Capture complete or ESC pressed.")
        break

cap.release()
cv2.destroyAllWindows()
