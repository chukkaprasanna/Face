import cv2
import os
import time

# ----- CONFIG -----
SNAPSHOT_DIR = "snapshots"   # folder to save face images
CAMERA_INDEX = 0             # change to 1 or 2 if your webcam is not 0
SCALE_FACTOR = 1.1           # face.detectMultiScale parameter
MIN_NEIGHBORS = 5            # face.detectMultiScale parameter
MIN_SIZE = (60, 60)          # minimum face size

# Create snapshots folder if not exists
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# Load the Haar cascade for face detection
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    raise IOError("Could not load Haar cascade xml. Check OpenCV installation or the path.")

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise IOError(f"Cannot open webcam (index={CAMERA_INDEX}). Try a different index.")

print("Press 'q' to quit, 's' to save snapshot of the detected faces.")

count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam. Exiting.")
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_SIZE
        )

        # Draw rectangles and labels
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Face {i+1}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Face Detector - Press 's' to save", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):   # quit
            break
        elif key == ord('s'): # save snapshot(s)
            timestamp = int(time.time())
            if len(faces) == 0:
                # Save whole frame if no faces detected
                filename = os.path.join(SNAPSHOT_DIR, f"frame_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved full frame: {filename}")
            else:
                # Save each detected face separately (cropped)
                for j, (x, y, w, h) in enumerate(faces):
                    face_img = frame[y:y+h, x:x+w]
                    filename = os.path.join(SNAPSHOT_DIR, f"face_{timestamp}_{j+1}.jpg")
                    cv2.imwrite(filename, face_img)
                    print(f"Saved face {j+1}: {filename}")
            count += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program ended. Snapshots saved:", os.path.abspath(SNAPSHOT_DIR))
