# Step 1: Import Libraries
import cv2
import mediapipe as mp

# Step 2: Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Step 3: Initialize Video Capture
cap = cv2.VideoCapture(0)

# Step 4: Capture and Process Each Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark coordinates
            landmark_list = []
            h, w, c = frame.shape
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])

            # Ensure all landmarks are available
            if len(landmark_list) == 21:
                gesture = None
                gesture2 = None

                # --- INDEX FINGER UP (This is my first ML project) ---
                if (landmark_list[8][1] < landmark_list[6][1] and  # Index up
                    landmark_list[12][1] > landmark_list[10][1] and
                    landmark_list[16][1] > landmark_list[14][1] and
                    landmark_list[20][1] > landmark_list[18][1]):
                    gesture = "THIS IS MY"
                    gesture2 = "FIRST ML PROJECT"

                # --- THUMB UP (Thanks) ---
                elif (landmark_list[4][1] < landmark_list[3][1] and  # Thumb up
                      landmark_list[8][1] > landmark_list[6][1] and
                      landmark_list[12][1] > landmark_list[10][1] and
                      landmark_list[16][1] > landmark_list[14][1] and
                      landmark_list[20][1] > landmark_list[18][1]):
                    gesture = "THANKS"

                # --- Display Text with Auto-Resizing Box ---
                if gesture:
                    # Measure text sizes
                    (text_w1, text_h1), _ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    box_w = text_w1 + 40
                    box_h = text_h1 + 50

                    if gesture2:
                        (text_w2, text_h2), _ = cv2.getTextSize(gesture2, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        box_w = max(box_w, text_w2 + 40)
                        box_h += text_h2 + 20  # increase height for second line

                    # Box position
                    x, y = landmark_list[0][0] - 150, landmark_list[0][1] - 130
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
                    alpha = 0.5
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # Draw text
                    cv2.putText(frame, gesture, (x + 20, y + 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
                    if gesture2:
                        cv2.putText(frame, gesture2, (x + 20, y + 95),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    # Step 5: Display the Frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 6: Release Resources
cap.release()
cv2.destroyAllWindows()
