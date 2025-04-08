import cv2
import mediapipe as mp
import math

# Funkcja obliczająca odległość między dwoma punktami
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detect_smile_and_open_mouth(frame):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Próg otwartych ust i uśmiechu
    THRESHOLD_OPEN = 0.05  
    THRESHOLD_SMILE_RATIO = 0.40  
    
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        
        face_landmarks = results.multi_face_landmarks[0]

        
        top_lip = face_landmarks.landmark[13]
        bottom_lip = face_landmarks.landmark[14]
        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]
        
        # Punkty dla szerokości twarzy (skronie) wykorzystane do skalowania odległości
        left_cheek = face_landmarks.landmark[234]
        right_cheek = face_landmarks.landmark[454]

        # Obliczanie odległości
        open_dist = distance(top_lip, bottom_lip)  # Otwarte usta (odległość pionowa)
        face_width = distance(left_cheek, right_cheek)  # Szerokość twarzy
        mouth_width = distance(left_mouth, right_mouth)  # Szerokość ust

        
        smile_ratio = mouth_width / face_width  # Proporcja szerokości ust do szerokości twarzy

        # Detekcja: Usta otwarte i uśmiech
        usta_otwarte = open_dist > THRESHOLD_OPEN  
        usmiech = smile_ratio > THRESHOLD_SMILE_RATIO  

        
        return usta_otwarte, usmiech
    
    return False, False

# Przykładowe użycie w głównym programie
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    usta_otwarte, usmiech = detect_smile_and_open_mouth(frame)

    print(f'Usta otwarte: {usta_otwarte} | Uśmiech: {usmiech}')

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
