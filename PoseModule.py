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
        self.model_complexity = 1  # Modèle de complexité minimale fixé à 1
        self.max_complexity = 2    # Modèle de complexité maximale fixé à 2
        self.min_complexity = 1    # Valeur minimale de complexité
        self.pose = None
        self.updatePoseModel()
        self.lmList = []

        # Détection de visage et de mains (pour informations supplémentaires)
        self.mpFace = mp.solutions.face_detection
        self.face_detection = self.mpFace.FaceDetection(min_detection_confidence=self.detectionCon)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=2,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)

    def updatePoseModel(self):
        """Créer un nouveau modèle de pose avec la complexité actuelle"""
        print(f"Création d'un nouveau modèle avec complexité: {self.model_complexity}")
        self.pose = mp.solutions.pose.Pose(static_image_mode=self.mode,
                                           model_complexity=self.model_complexity,
                                           smooth_landmarks=self.smooth,
                                           enable_segmentation=self.upBody,
                                           min_detection_confidence=self.detectionCon,
                                           min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        """Applique la détection de pose"""
        if img is None:
            print("Erreur : l'image est None. Assurez-vous que l'image a été correctement chargée.")
            return None

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(img, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """Récupérer la position des landmarks détectés dans l'image"""
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z  # Coordonnées 3D (x, y, z)
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def tryDifferentComplexities(self, img):
        """Essaie différentes valeurs de complexité jusqu'à trouver la meilleure"""
        for complexity in range(self.min_complexity, self.max_complexity + 1):
            self.model_complexity = complexity
            self.updatePoseModel()  # Recréer le modèle avec la nouvelle complexité

            img = self.findPose(img, draw=False)
            self.findPosition(img)
            if self.lmList:
                print(f"Personne détectée avec une complexité de {self.model_complexity}")
                return True, img
        print("Aucune personne détectée après avoir testé toutes les complexités.")
        return False, img

    def getUpperBodyLandmarks(self, img):
        """Récupérer les landmarks spécifiques à la tête, aux épaules, au dos et aux bras"""
        landmarks_of_interest = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # Landmarks du haut du corps
        self.lmList = []
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if id in landmarks_of_interest:
                    h, w, c = img.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z  # Coordonnées 3D
                    self.lmList.append([id, cx, cy, cz])
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # Dessiner le landmark sur l'image
        return self.lmList

    # Nouvelle fonction pour détecter les actions techniques par contours
    def detect_actions_from_movement(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 30, 100)  # Ajustement du seuil de Canny
        _, thresh_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_contours = []
        nb_actions = 0
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > 500:  # Réduction du seuil pour détecter plus de mouvements
                nb_actions += 1
                detected_contours.append(contour)
                # Dessiner les contours sur l'image pour visualisation
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        print(f"Nombre d'actions techniques détectées (Contours) : {nb_actions}")
        return nb_actions, detected_contours

    # Fonction pour la détection des actions techniques basée sur la soustraction d'arrière-plan
    def detect_actions_with_bg_subtraction(self, image, bg_subtractor):
        fg_mask = bg_subtractor.apply(image)

        _, thresh_image = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_contours = []
        nb_actions = 0
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Ajuster la taille minimale des objets
                nb_actions += 1
                detected_contours.append(contour)
                cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)  # Différencier par la couleur (bleu)

        print(f"Nombre d'actions techniques détectées (Soustraction d'Arrière-Plan) : {nb_actions}")
        return nb_actions, detected_contours

    # Fonction pour éviter les doublons (fusionner les contours similaires)
    def remove_duplicate_actions(self, contours1, contours2):
        final_contours = contours1.copy()

        for contour2 in contours2:
            duplicate = False
            for contour1 in contours1:
                # Vérifier la proximité et la taille des contours pour éviter les doublons
                distance = cv2.pointPolygonTest(contour1, tuple(contour2[0][0]), True)
                if abs(distance) < 50:  # Si les contours sont très proches
                    duplicate = True
                    break

            if not duplicate:
                final_contours.append(contour2)

        return final_contours

    # Fonction pour définir une zone d'intérêt (ROI) devant les mains
    def get_roi_in_front_of_hands(self, landmarks, margin=70, forward_margin=120):
        """
        Définir une zone englobante devant les mains (poignets).
        margin : marge autour des mains
        forward_margin : distance supplémentaire vers l'avant des mains
        """
        # Landmarks des coudes et poignets
        arm_landmarks_ids = [13, 14, 15, 16]

        # Récupérer les coordonnées X, Y des coudes et poignets
        x_coords = [lm[1] for lm in landmarks if lm[0] in arm_landmarks_ids]
        y_coords = [lm[2] for lm in landmarks if lm[0] in arm_landmarks_ids]

        # Définir les limites de la zone englobante (bounding box) autour des bras et mains
        x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
        y_min, y_max = min(y_coords) - margin, max(y_coords) + margin

        # Étendre la zone vers l'avant des mains en ajoutant un forward_margin à x_max
        x_max += forward_margin

        return x_min, y_min, x_max, y_max
