import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import threading
from collections import deque

# ──────────────────────────────────────────────
# Non-blocking beep
# ──────────────────────────────────────────────
def beep_async(frequency=1000, duration=500):
    def _beep():
        try:
            import winsound
            winsound.Beep(frequency, duration)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(landmarks, start, end):
    """Generic EAR for a set of 6 eye landmarks."""
    pts = landmarks[start:end]
    up   = compute(pts[1], pts[5]) + compute(pts[2], pts[4])
    down = compute(pts[0], pts[3])
    return up / (2.0 * down)

def mouth_aspect_ratio(mouth):
    A = compute(mouth[2], mouth[10])
    B = compute(mouth[4], mouth[8])
    C = compute(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SCALE          = 0.75     # Higher = better detection, slightly slower
UPSAMPLE       = 1        # 1 = more reliable face finding vs 0
DETECT_EVERY   = 2        # Run dlib detector every N frames (reuse between)
SMOOTH_WINDOW  = 5        # Frames to average EAR over (reduces flicker)

EAR_THRESH_CLOSED  = 0.21
EAR_THRESH_DROWSY  = 0.25
MAR_THRESH_YAWN    = 0.65

SLEEP_FRAMES   = 20       # Consecutive closed-eye frames → SLEEPING
DROWSY_FRAMES  = 8
ACTIVE_FRAMES  = 6
YAWN_FRAMES    = 25

BEEP_COOLDOWN  = 2.0


# ──────────────────────────────────────────────
# Init
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# State
sleep = drowsy = active = yawn_count = 0
status = ""
color  = (0, 0, 0)

night_mode     = False
prev_time      = 0
last_beep_time = 0
frame_count    = 0

# Cached detection results (reused on skip frames)
cached_faces     = []
cached_landmarks = None   # landmarks in small-frame coords

# Smoothing buffers for left/right EAR
left_ear_buf  = deque(maxlen=SMOOTH_WINDOW)
right_ear_buf = deque(maxlen=SMOOTH_WINDOW)


# ──────────────────────────────────────────────
# Classify smoothed EAR
# ──────────────────────────────────────────────
def classify_ear(ear_avg):
    if ear_avg > EAR_THRESH_DROWSY:
        return 2   # open
    elif ear_avg > EAR_THRESH_CLOSED:
        return 1   # drowsy
    else:
        return 0   # closed


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # ── Prepare small gray frame ─────────────
    small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if night_mode:
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Run detector only every DETECT_EVERY frames ──
    if frame_count % DETECT_EVERY == 0:
        cached_faces = detector(gray, UPSAMPLE)

    faces      = cached_faces
    face_frame = frame.copy()

    if len(faces) == 0:
        status = "No Face Detected"
        color  = (0, 255, 255)
        # Drain buffers so stale EAR doesn't linger
        left_ear_buf.clear()
        right_ear_buf.clear()

    for face in faces:
        # ── Bounding box scaled back to original ──
        x1 = int(face.left()  / SCALE)
        y1 = int(face.top()   / SCALE)
        x2 = int(face.right() / SCALE)
        y2 = int(face.bottom()/ SCALE)
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Landmarks (small frame coords) ───────
        shape = predictor(gray, face)
        lm    = face_utils.shape_to_np(shape)          # small coords
        lm_orig = (lm / SCALE).astype(int)             # display coords

        # ── Smooth EAR ───────────────────────────
        left_ear  = eye_aspect_ratio(lm, 36, 42)
        right_ear = eye_aspect_ratio(lm, 42, 48)
        left_ear_buf.append(left_ear)
        right_ear_buf.append(right_ear)

        avg_left  = np.mean(left_ear_buf)
        avg_right = np.mean(right_ear_buf)

        left_blink  = classify_ear(avg_left)
        right_blink = classify_ear(avg_right)

        # ── MAR ──────────────────────────────────
        mouth = lm[48:68]
        mar   = mouth_aspect_ratio(mouth)

        # ── State machine ────────────────────────
        if left_blink == 0 and right_blink == 0:
            sleep  += 1
            drowsy  = 0
            active  = 0
            if sleep > SLEEP_FRAMES:
                status = "SLEEPING !!!"
                color  = (255, 0, 0)
                if current_time - last_beep_time > BEEP_COOLDOWN:
                    beep_async(1000, 500)
                    last_beep_time = current_time

        elif left_blink == 1 or right_blink == 1:
            sleep  = 0
            active = 0
            drowsy += 1
            if drowsy > DROWSY_FRAMES:
                status = "Drowsy !"
                color  = (0, 0, 255)

        else:
            drowsy = 0
            sleep  = 0
            active += 1
            if active > ACTIVE_FRAMES:
                status = "Active :)"
                color  = (0, 255, 0)

        # ── Yawn ─────────────────────────────────
        if mar > MAR_THRESH_YAWN:
            yawn_count += 1
        else:
            yawn_count = 0
        if yawn_count > YAWN_FRAMES:
            status = "Yawning!"
            color  = (0, 165, 255)

        # ── Draw EAR values on frame ─────────────
        cv2.putText(frame, f"L-EAR: {avg_left:.2f}",
                    (430, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"R-EAR: {avg_right:.2f}",
                    (430, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}",
                    (430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # ── Draw landmarks ───────────────────────
        for (x, y) in lm_orig:
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # ── HUD (drawn once, outside face loop) ──
    cv2.putText(frame, status,
                (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20,  50),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    mode_text = "Night Mode ON" if night_mode else "Normal Mode"
    cv2.putText(frame, mode_text,
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "N = toggle night | ESC = quit",
                (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Drowsiness Detector", frame)
    cv2.imshow("Landmarks", face_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        night_mode = not night_mode
        print(f"Night mode: {'ON' if night_mode else 'OFF'}")
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()