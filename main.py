import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Screen size
screen_width, screen_height = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Cursor smoothing
prev_x, prev_y = 0, 0
smooth = 5

# FPS
prev_time = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    gesture = "No Hand"

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Important landmarks
        thumb = hand.landmark[4]
        index = hand.landmark[8]
        middle = hand.landmark[12]

        # -------------------------
        # CURSOR CONTROL
        # -------------------------

        target_x = int(index.x * screen_width)
        target_y = int(index.y * screen_height)

        curr_x = prev_x + (target_x - prev_x) / smooth
        curr_y = prev_y + (target_y - prev_y) / smooth

        pyautogui.moveTo(curr_x, curr_y)

        prev_x = curr_x
        prev_y = curr_y

        # -------------------------
        # THUMB + INDEX DISTANCE
        # -------------------------

        distance = math.sqrt(
            (thumb.x - index.x) ** 2 +
            (thumb.y - index.y) ** 2
        )

        # Pinch = Left Click
        if distance < 0.045:

            gesture = "LEFT CLICK"

            pyautogui.click()

            time.sleep(0.3)

        # -------------------------
        # TWO FINGER
        # -------------------------

        index_middle_distance = math.sqrt(
            (index.x - middle.x) ** 2 +
            (index.y - middle.y) ** 2
        )

        if index_middle_distance < 0.08:

            gesture = "TWO FINGERS"

        else:

            gesture = "MOVE CURSOR"

    # -------------------------
    # FPS
    # -------------------------

    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

    prev_time = current_time

    # -------------------------
    # UI
    # -------------------------

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Hand Gesture PC Controller",
        frame
    )

    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()