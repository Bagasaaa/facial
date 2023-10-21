import os.path

import cv2
import math
import json
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import gdown

from webcolors import hex_to_rgb

from ultralytics import YOLO

NUM_FACE = 1

if not os.path.isfile("models/lugas_ganteng_kedua_setelah_mas_bagas.pt"):
    gdown.download(id="1okJRBft8sAklYXqiP0eMfU-Lm_VA6p4C", output="models/lugas_ganteng_kedua_setelah_mas_bagas.pt")

if not os.path.isfile("models/model_kacamata_lugastyan.pt"):
    gdown.download(id="1lXt0TUCzFmF3aDGxEdyBxwI2ZDyXh1Rp", output="models/model_kacamata_lugastyan.pt")

with open("assets/json/color.json", "r", encoding="utf-8") as file:
    color_data = json.load(file)
    file.close()

class FaceLandMarks:
    def __init__(
        self, staticMode=True, maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.5
    ):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode, max_num_faces=self.maxFace,
            min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceLandmark(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                # if draw:
                #     self.mpDraw.draw_landmarks(
                #         img,
                #         faceLms,
                #         self.mpFaceMesh.FACE_CONNECTIONS, #FACEMESH_CONTOURS,FACE_CONNECTIONS
                #         self.drawSpec,
                #         self.drawSpec,
                #     )

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                    # Menampilkan nomor di setiap landmark
                    # cv2.putText(
                    #     img,
                    #     str(id),
                    #     (x, y),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.3,
                    #     (0, 255, 0),
                    #     1,
                    # )
                faces.append(face)
        return img, faces


def extract_color_roi(image, landmarks):
    colors = []
    for landmark in landmarks:
        x, y = landmark
        roi = image[
            y - 5 : y + 5, x - 5 : x + 5
        ]  # Customize the ROI size here (50x50 pixels)
        color_rgb = np.mean(roi, axis=(0, 1)).astype(int)[
            ::-1
        ]  # Convert BGR to RGB format and calculate the mean
        colors.append(color_rgb)
    return colors


def plot_color_plot(colors_dict):
    rows = len(colors_dict)
    cols = max(len(colors) for colors in colors_dict.values())
    fig, ax = plt.subplots(rows, cols, figsize=(cols, rows))

    for row, (key, colors) in enumerate(colors_dict.items()):
        for col, color in enumerate(colors):
            color_rgb = [c / 255.0 for c in color]  # Normalize RGB values to [0, 1]
            ax[row, col].imshow([[color_rgb]], aspect="auto")
            ax[row, col].axis("off")
            ax[row, col].set_title(key)

    plt.tight_layout()
    plt.show()


def find_nearest_color_name(hex_color):
    target_rgb = hex_to_rgb(hex_color)
    nearest_color = None
    min_distance = float("inf")

    for color in color_data:
        color_rgb = hex_to_rgb(color["hex"])
        distance = calculate_distance(target_rgb, color_rgb)
        if distance < min_distance:
            min_distance = distance
            nearest_color = color["name"]

    return nearest_color


def calculate_distance(color1, color2):
    # Calculate Euclidean distance between two colors in RGB space
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def classify_makeup_colors(image_path):
    model = YOLO("models/lugas_ganteng_kedua_setelah_mas_bagas.pt")
    model.fuse()
    predict_raw = model.predict(image_path, save=True, project="./result/", exist_ok=True, conf = 0.2)
    names = model.names
    class_names = []  # List to store predicted class names
    for predict in predict_raw:
        for pre in predict.boxes.cls:
            class_name = names[int(pre)]
            class_names.append(class_name)
    return class_names

def classify_glasses_or_no(image_path):
    model = YOLO("models/model_kacamata_lugastyan.pt")
    model.fuse()
    predict_raw_glasses = model.predict(image_path)
    names = model.names
    class_names_glasses = []
    for predict in predict_raw_glasses:
        for pre in predict.boxes.cls:
            class_name = names[int(pre)]
            class_names_glasses.append(class_name)
    return class_names_glasses