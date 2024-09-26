import cv2
from PoseModule import poseDetector
import numpy as np
import os

detector = poseDetector(upBody=True)

class ActionHistory:
    """Classe pour suivre l'historique des actions sur plusieurs frames."""
    def __init__(self, history_length=5):
        self.history = [False] * history_length

    def update_history(self, action_detected):
        """Mise à jour de l'historique des actions."""
        self.history.pop(0)
        self.history.append(action_detected)

    def action_persistent(self, threshold=3):
        """Détecter si une action est persistante sur plusieurs frames."""
        return sum(self.history) >= threshold

action_history = ActionHistory(history_length=5)

def detect_combined_actions(image_path):
    """Détecte la personne et les actions techniques combinées dans une image."""
    print("---- Détection de la personne avec complexité ----")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image à partir de {image_path}")
        return

    person_detected, image_with_person = detector.tryDifferentComplexities(image)
    
    if not person_detected:
        print("Aucune personne détectée.")
        cv2.putText(image, "Aucune personne detectee", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Resultat", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    upper_body_landmarks = detector.getUpperBodyLandmarks(image_with_person)
    print(f"Landmarks des bras et des mains détectés : {upper_body_landmarks}")

    # Suivi des mouvements des landmarks
    movements = detector.track_landmarks(detector.previous_landmarks, upper_body_landmarks)
    significant_movement = detector.detect_significant_movement(movements)
    detector.previous_landmarks = upper_body_landmarks

    x_min, y_min, x_max, y_max = detector.get_roi_in_front_of_hands(upper_body_landmarks, margin=70, forward_margin=120)
    print(f"Zone englobante devant les mains : ({x_min}, {y_min}) - ({x_max}, {y_max})")

    roi = image_with_person[y_min:y_max, x_min:x_max]
    actions_from_movement, contours_movement = detector.detect_actions_from_movement(roi)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    actions_from_bg_subtraction, contours_bg = detector.detect_actions_with_bg_subtraction(roi, bg_subtractor)

    for contour in contours_movement:
        cv2.drawContours(image_with_person, [contour], -1, (255, 0, 0), 2)

    combined_contours = detector.remove_duplicate_actions(contours_movement, contours_bg)
    total_actions = len(combined_contours)
    print(f"\nTotal d'actions techniques détectées dans la zone devant les mains : {total_actions}")

    cv2.rectangle(image_with_person, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if total_actions == 0 and not significant_movement:
        print("Il ne se passe rien.")
        cv2.putText(image_with_person, "Il ne se passe rien", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image_with_person, f"Actions detectees : {total_actions}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(image_with_person, "Personne detectee", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Resultat", image_with_person)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Vérification de l'existence de l'image avant exécution
image_path = "/chemin/vers/ton/image.jpg"
if not os.path.exists(image_path):
    print(f"L'image spécifiée n'existe pas : {image_path}")
else:
    detect_combined_actions(image_path)
