import cv2
import mediapipe as mp
import numpy as np

class poseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = 1
        self.max_complexity = 2
        self.min_complexity = 1
        self.pose = None
        self.updatePoseModel()
        self.lmList = []
        self.previous_landmarks = []  # Pour le suivi des mouvements entre frames

    def updatePoseModel(self):
        """Crée un nouveau modèle de détection de pose en fonction de la complexité actuelle."""
        print(f"Création d'un nouveau modèle avec complexité: {self.model_complexity}")
        self.pose = mp.solutions.pose.Pose(static_image_mode=self.mode,
                                           model_complexity=self.model_complexity,
                                           smooth_landmarks=self.smooth,
                                           enable_segmentation=self.upBody,
                                           min_detection_confidence=self.detectionCon,
                                           min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        """Applique la détection de pose sur l'image."""
        if img is None:
            print("Erreur : l'image est None.")
            return None
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(img, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """Récupère les positions des landmarks détectés."""
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def track_landmarks(self, previous_landmarks, current_landmarks):
        """Suit les mouvements des landmarks entre deux frames."""
        movements = []
        for prev, curr in zip(previous_landmarks, current_landmarks):
            if prev and curr:
                distance = np.linalg.norm(np.array(prev[1:3]) - np.array(curr[1:3]))
                movements.append(distance)
        return movements

    def detect_significant_movement(self, movements, threshold=10):
        """Détecte les mouvements significatifs au-dessus d'un seuil donné."""
        return any(movement > threshold for movement in movements)

    def detect_actions_from_movement(self, image):
        """Détecte les actions techniques en utilisant les contours d'objets en mouvement."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 30, 100)
        _, thresh_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_contours = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > 500:
                detected_contours.append(contour)
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        return len(detected_contours), detected_contours

    def filter_small_movements(self, contours, min_area=500):
        """Filtre les petits mouvements ne correspondant pas à des actions significatives."""
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                filtered_contours.append(contour)
        return filtered_contours

    def detect_actions_with_bg_subtraction(self, image, bg_subtractor):
        """Détecte les actions techniques en utilisant la soustraction d'arrière-plan."""
        fg_mask = bg_subtractor.apply(image)
        _, thresh_image = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.filter_small_movements(contours, min_area=500)
        for contour in filtered_contours:
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
        return len(filtered_contours), filtered_contours

    def remove_duplicate_actions(self, contours1, contours2):
        """Fusionne les contours similaires pour éviter les doublons."""
        final_contours = contours1.copy()
        for contour2 in contours2:
            duplicate = False
            for contour1 in contours1:
                distance = cv2.pointPolygonTest(contour1, tuple(contour2[0][0]), True)
                if abs(distance) < 50:
                    duplicate = True
                    break
            if not duplicate:
                final_contours.append(contour2)
        return final_contours

    def get_roi_in_front_of_hands(self, landmarks, margin=70, forward_margin=120):
        """Définit une zone d'intérêt (ROI) devant les mains pour la détection des actions."""
        arm_landmarks_ids = [13, 14, 15, 16]
        x_coords = [lm[1] for lm in landmarks if lm[0] in arm_landmarks_ids]
        y_coords = [lm[2] for lm in landmarks if lm[0] in arm_landmarks_ids]
        x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
        y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
        x_max += forward_margin
        return x_min, y_min, x_max, y_max
