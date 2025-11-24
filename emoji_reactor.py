import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Konfigurasi dan Ambang Batas ---
SMILE_THRESHOLD = 0.35
SURPRISE_THRESHOLD = 0.6  # Ambang batas untuk mulut terbuka (kaget)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# --- Memuat gambar emoji kera ---
try:
    # Emoji Kera
    kera_berpikir_emoji = cv2.imread("kera_berpikir.jpg")
    kera_ide_emoji = cv2.imread("kera_ide.jpg")
    kera_kaya_emoji = cv2.imread("kera_kaya.jpg")
    kera_kaget_emoji = cv2.imread("kera_kaget.jpg")

    # Periksa apakah semua file berhasil dimuat
    image_files = {
        "kera_berpikir.jpg": kera_berpikir_emoji, "kera_ide.jpg": kera_ide_emoji,
        "kera_kaya.jpg": kera_kaya_emoji, "kera_kaget.jpg": kera_kaget_emoji
    }
    for filename, img in image_files.items():
        if img is None:
            raise FileNotFoundError(f"{filename} not found or could not be read.")

    # Resize semua emoji
    for key, img in image_files.items():
        image_files[key] = cv2.resize(img, EMOJI_WINDOW_SIZE)
    
    # Ekstrak kembali ke variabel masing-masing setelah di-resize
    kera_berpikir_emoji = image_files["kera_berpikir.jpg"]
    kera_ide_emoji = image_files["kera_ide.jpg"]
    kera_kaya_emoji = image_files["kera_kaya.jpg"]
    kera_kaget_emoji = image_files["kera_kaget.jpg"]

except Exception as e:
    print(f"Error loading emoji images! Details: {e}")
    print("\nPastikan file-file berikut ada di folder yang sama:")
    print("- kera_berpikir.jpg, kera_ide.jpg, kera_kaya.jpg, kera_kaget.jpg")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[1], EMOJI_WINDOW_SIZE[0], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Setup windows
cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")

# Inisialisasi MediaPipe dengan 'with'
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Proses semua model
        results_face = face_mesh.process(image_rgb)
        results_hands = hands.process(image_rgb)

        # Inisialisasi state default (tidak ada state)
        current_state = None
        is_smiling = False

        # --- Logika Deteksi dengan Prioritas ---
        
        # Deteksi Ekspresi Wajah (diperlukan untuk gestur gabungan)
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0]

            # --- TAMBAHAN: GAMBAR BINGKAI WAJAH ---
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
            # Tambahkan sedikit padding agar bingkai tidak terlalu ketat
            padding = 20
            cv2.rectangle(frame, (x_min - padding, y_min - padding), (x_max + padding, y_max + padding), (0, 255, 0), 2)
            # -----------------------------------------

            # Kalkulasi Rasio Aspek Mulut
            p_upper_lip = face_landmarks.landmark[13]
            p_lower_lip = face_landmarks.landmark[14]
            p_left_corner = face_landmarks.landmark[291]
            p_right_corner = face_landmarks.landmark[61]
            mouth_width = math.hypot(p_right_corner.x - p_left_corner.x, p_right_corner.y - p_left_corner.y)
            mouth_height = math.hypot(p_lower_lip.x - p_upper_lip.x, p_lower_lip.y - p_upper_lip.y)
            if mouth_width > 0:
                mouth_aspect_ratio = mouth_height / mouth_width
                if mouth_aspect_ratio > SURPRISE_THRESHOLD:
                    current_state = "KAGET"
                elif mouth_aspect_ratio > SMILE_THRESHOLD:
                    is_smiling = True
        
        # Prioritas 1: Deteksi Gestur Tangan (jika tidak sedang kaget)
        if current_state != "KAGET" and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Cek Jari Telunjuk ke Atas (Hanya untuk "Ide")
                index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                
                if index_tip_y < index_pip_y and middle_tip_y > index_pip_y:
                    current_state = "KERA_IDE" # Selalu menjadi "ide"
                    break

                # Cek Pose Berpikir (tangan di dekat dagu)
                if results_face.multi_face_landmarks:
                    chin_landmark = face_landmarks.landmark[152]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    distance = math.hypot(index_tip.x - chin_landmark.x, index_tip.y - chin_landmark.y)
                    if distance < 0.1:
                        current_state = "KERA_BERPIKIR"
                        break

        # Prioritas 2: Deteksi Senyum (jika tidak ada gestur lain terdeteksi)
        if current_state is None and is_smiling:
            current_state = "KERA_KAYA"

        # --- Pilih emoji berdasarkan state akhir ---
        if current_state == "KAGET":
            emoji_to_display = kera_kaget_emoji
            emoji_name = "😮 Kaget"
        elif current_state == "KERA_BERPIKIR":
            emoji_to_display = kera_berpikir_emoji
            emoji_name = "🤔 Berpikir"
        elif current_state == "KERA_IDE":
            emoji_to_display = kera_ide_emoji
            emoji_name = "💡 Punya Ide"
        elif current_state == "KERA_KAYA":
            emoji_to_display = kera_kaya_emoji
            emoji_name = "🤑 Kaya!"
        else: # Jika tidak ada state yang terdeteksi
            emoji_to_display = blank_emoji
            emoji_name = "..."

        # Tampilkan frame dan emoji
        # Kita menggambar di 'frame' asli, lalu me-resize-nya.
        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.putText(camera_frame_resized, f'STATE: {emoji_name}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()