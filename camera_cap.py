import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    # Visualizing
    cv2.imshow('Mediapipe Feed', frame)

    # If q is pressed break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
