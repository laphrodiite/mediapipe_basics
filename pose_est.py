import cv2
import mediapipe as mp
import numpy as np
from angle_func import calculate_angle

# This will give us our drawing utilities to visualize
mp_drawing = mp.solutions.drawing_utils
# This will give us the pose estimation model
mp_pose = mp.solutions.pose


# Taking video feed
cap = cv2.VideoCapture(0)
## Setup mediapipe instance (high accuracy -> bigger value (%))
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        # Reading current feed
        ret, frame = cap.read()
        
        ## Recolor image (cv2 default is BGR but MediaPipe work)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        ## Making detection
        results = pose.process(image)
        
        ## Recoloring 
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ### Extract landmarks
        """
        Landmarks is a list of all the available landmarks that can be tracked 
        and then their coordinated change according to the movement.
        In order to map them we do
        for lndmrk in mp_pose.PoseLandmark:
            print(lndmrk)
        To get the coordinates of one
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        """
        try:
            landmarks = results.pose_landmarks.landmark
            ### Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            ### Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            ### Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [1280, 720]).astype(int),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AAA)
                        )
        except:
            pass


        ## Rendering detections (drawing results on image)
        """
        mp_pose.POSE_CONNECTIONS : All available points on human body
        Second line is for the dots and third is for the lines
        """
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2),
                                 mp_drawing.DrawingSpec(color=(42,132,234), thickness = 2, circle_radius = 2))
       
        # Visualizing ## changing frame to image
        cv2.imshow('Mediapipe Feed', image)

        # If q is pressed break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
