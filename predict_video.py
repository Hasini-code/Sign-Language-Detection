import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("sign_language_model.h5")

# Update labels according to folder names
labels = ["NO", "YES"]

# Load test video
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    predicted_label = labels[np.argmax(pred)]
    predictions.append(np.argmax(pred))

    cv2.putText(frame, predicted_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Final decision
final_prediction = labels[max(set(predictions), key=predictions.count)]
print("Final Prediction:", final_prediction)
