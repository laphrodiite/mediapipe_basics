import cv2
import mediapipe as mp
import time

# ----------------------------------------------------------------------------------------------
# MediaPipe Hand module initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
"""
Inside the Hands() function:
    def __innit__ (self,
                    static_image_mode=False, (1)
                    max_num_hands=2, (2)
                    min_detection_confidence=0.5, (3)
                    min_tracking_confidence=0.5) (4)

(1):    False means sometimes it will detect and sometimes will track based on confidence level
        if True it will always detect which makes it slow
(2):    Maximum number of hands
(3):    50%
(4):    50% 
"""

# ----------------------------------------------------------------------------------------------
# Camera usage
cap = cv2.VideoCapture(0)

# Initialization for fps
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    success, image = cap.read()
    ## Recolor image (cv2 default is BGR but MediaPipe work)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image)
    # To see if it tracks anything
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        # For each hand landmark
        for hand_lms in results.multi_hand_landmarks:
            # We will get the id number for the joint and landmark (coordinates) for each hand
            for id, lm in enumerate(hand_lms.landmark):
                """
                print(id,lm)
                This gives you a full list of the 
                """
                # In order to get the image coordinates we need to multiply by dimensions
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                # If we want to see a specific point:
                if id == 8:
                    cv2.circle(imag)
                

                # If you just want the points then stop at the second arguement
                mp_draw.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS)

    ## Recoloring 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # -----------------------------------------------------------------------
    # Display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time 
    fps = str(int(fps)) 
    image = cv2.flip(image,1)
    cv2.putText(image, fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA) 
    # ------------------------------------------------------------------------

    # Visualizing
    cv2.imshow('Mediapipe Feed',image)

    # If q is pressed break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
