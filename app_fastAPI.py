import os
import cv2
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from webcolors import rgb_to_hex
from assets.cascade import cascade_mantap
from enum import Enum
from pydantic import BaseModel

from assets.facial_landmarks import (
    FaceLandMarks,
    extract_color_roi,
    find_nearest_color_name,
    classify_makeup_colors,
    classify_glasses_or_no
)

app = FastAPI()

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
UPLOAD_FOLDER = './user_photo/'


def allowed_photo(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class TujuanMakeup(str, Enum):
    Halloween = "Halloween"
    Konser = "Konser"
    Cosplay = "Cosplay"
    Wedding = "Wedding"
    Graduation = "Graduation"
    PromNight = "PromNight"
    Party = "Party"
    Sekolah = "Sekolah"
    Kuliah = "Kuliah"
    Kerja = "Kerja"
    Travelling = "Travelling"

class TujuanMakeup2(str, Enum):
    Halloween = "Halloween"
    Konser = "Konser"
    Cosplay = "Cosplay"
    Wedding = "Wedding"
    Graduation = "Graduation"
    PromNight = "PromNight"
    Party = "Party"
    Sekolah = "Sekolah/Kuliah"
    Kerja = "Kerja"
    Travelling = "Travelling"

class MakeupResponse(BaseModel):
    average_colors_rgb: dict
    average_colors_hex: dict
    average_colors_names: dict
    Blush: dict
    Lipstick: dict
    Eyeshadow: dict
    Foundation: dict
    uploaded_image: str
    kategori_makeup: str
    kategori_makeup_en: str
    deskripsi_makeup: str
    deskripsi_makeup_en: str
    rekomendasi_tema_makeup: list
    pemilihan_warna: str
    pemilihan_warna_en: str
    tips_makeup_lain: str
    tips_makeup_lain_en: str
    kecocokan_makeup: bool
    rekomendasi_makeup_eyeshadow: list
    rekomendasi_makeup_blushon: list
    rekomendasi_makeup_lipstick: list
    rekomendasi_makeup_overall: list

class MakeupResponse(BaseModel):
    blush_color: str
    lipstick_color: str
    eyeshadow_color: str
    foundation_color: str
    description: str
    
@app.post("/facialdetection/")
async def facialdetection(file: UploadFile, tujuan_makeup: TujuanMakeup = Form(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        if not allowed_photo(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")

        temp_filename_base = os.path.join(UPLOAD_FOLDER, "temp_image.jpg")

        with open(temp_filename_base, "wb") as temp_file:
            temp_file.write(file.file.read())

        temp_filename = cascade_mantap(temp_filename_base)

        if temp_filename != 0:
            img_raw = cv2.imread(temp_filename_base)
            detector = FaceLandMarks(staticMode=True)
            img, faces = detector.findFaceLandmark(img_raw)

            if len(faces) != 0:
                # Code for drawing and displaying landmarks
                for face in faces:
                    for landmark in face:
                        x, y = landmark

                # List landmark yang ingin diambil warna rata-ratanya
                pipi_landmarks = [50, 330, 205, 101]
                bibir_landmarks = [15, 16, 72, 11, 85, 180]
                eye_shadow_landmarks = [222, 223, 224, 225, 442, 443, 444, 445]
                foundation_landmarks = [5, 195, 197, 248, 3]

                # Convert NumPy arrays to Python lists
                pipi_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in pipi_landmarks])]
                bibir_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in bibir_landmarks])]
                eye_shadow_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in eye_shadow_landmarks])]
                foundation_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in foundation_landmarks])]

                # Calculate average color for each makeup component
                average_blush_color = np.mean(pipi_colors, axis=0).astype(int).tolist()
                average_lipstick_color = np.mean(bibir_colors, axis=0).astype(int).tolist()
                average_eyeshadow_color = np.mean(eye_shadow_colors, axis=0).astype(int).tolist()
                average_foundation_color = np.mean(foundation_colors, axis=0).astype(int).tolist()

                # Create a dictionary to store the average colors
                average_colors_dict = {
                    "Blush": average_blush_color,
                    "Lipstick": average_lipstick_color,
                    "Eyeshadow": average_eyeshadow_color,
                    "Foundation": average_foundation_color
                }
                
                # Get color names for each average color
                average_colors_names_dict = {}
                for key, color in average_colors_dict.items():
                    color_rgb = tuple(color)
                    print(color_rgb)
                    color_hex = rgb_to_hex(color_rgb)
                    print(color_hex)
                    color_name = find_nearest_color_name(color_hex)
                    average_colors_names_dict[key] = color_name

                blush_dict = {
                    "average_colors_rgb": average_colors_dict['Blush'],
                    "average_colors_hex": rgb_to_hex(average_colors_dict["Blush"]),
                    "average_colors_names": average_colors_names_dict['Blush']
                }

                print(blush_dict)

                lipstick_dict = {
                    "average_colors_rgb": average_colors_dict['Lipstick'],
                    "average_colors_hex": rgb_to_hex(average_colors_dict["Lipstick"]),
                    "average_colors_names": average_colors_names_dict['Lipstick']
                }

                print(lipstick_dict)

                eyeshadow_dict = {
                    "average_colors_rgb": average_colors_dict['Eyeshadow'],
                    "average_colors_hex": rgb_to_hex(average_colors_dict["Eyeshadow"]),
                    "average_colors_names": average_colors_names_dict['Eyeshadow']
                }

                print(eyeshadow_dict)

                foundation_dict = {
                    "average_colors_rgb": average_colors_dict['Foundation'],
                    "average_colors_hex": rgb_to_hex(average_colors_dict["Foundation"]),
                    "average_colors_names": average_colors_names_dict['Foundation']
                }

                print(foundation_dict)

                image_glasses_or_no = classify_glasses_or_no(temp_filename_base)
                image_glasses_or_no_str = str(image_glasses_or_no[0])
                print("#####################################")
                print(image_glasses_or_no_str)
                if image_glasses_or_no_str == 'no_glasses':
                    image_data_predicted = classify_makeup_colors(temp_filename_base)
                    image_data_predicted_true = str(image_data_predicted[0])
                    label_predicted = str(image_data_predicted[0]).replace("_", " ")
                    print(label_predicted)

                    # Load data from makeup_description.json
                    with open('assets/json/makeup_description.json', 'r') as makeup_description:
                        makeup_data = json.load(makeup_description)

                    # Get makeup description based on predicted label
                    specific_makeup_description = makeup_data.get(label_predicted)
                    print(specific_makeup_description)

                    rekomendasi_makeup_eyeshadows = []
                    rekomendasi_makeup_blushons = []
                    rekomendasi_makeup_lipsticks = []
                    rekomendasi_makeup_overalls = []

                    recommended_themes = []
                    for makeup_label, makeup_info in makeup_data.items():
                        makeup_label = makeup_label.replace(" ", "_")
                        if tujuan_makeup in makeup_info["tujuan_makeup"]:
                            recommended_themes.append(makeup_label)

                    if "no_makeup" in recommended_themes:
                        recommended_themes.remove("no_makeup")

                    if label_predicted == 'vintage makeup':
                        for i in range(1, 7):  # Loop dari 1 hingga 6
                            rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                        for i in range(1, 4):
                            rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{image_data_predicted_true}{i}.png"
                            rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                            rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                            rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)
                        
                    elif label_predicted == 'fantasy makeup':
                        for i in range(1, 5):
                            rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                        for i in range(1, 4):
                            rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{image_data_predicted_true}{i}.png"
                            rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                            rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                            rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)

                    elif label_predicted == 'no makeup':
                        for i in range(1, 4):
                            rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_casual_makeup{i}.jpg"
                            rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_casual_makeup{i}.jpg"
                            rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_casual_makeup{i}.jpg"
                            rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_casual_makeup{i}.png"
                            rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                            rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                            rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)
                            rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)

                    else:
                        for i in range(1, 4):  # Loop dari 1 hingga 4
                            rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{image_data_predicted_true}{i}.jpg"
                            rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{image_data_predicted_true}{i}.png"
                            rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                            rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                            rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                            rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)

                    # Get the specific makeup description based on tujuan_makeup
                    if label_predicted in makeup_data and tujuan_makeup in makeup_data[label_predicted]["tujuan_makeup"]:
                        specific_makeup_description = makeup_data[label_predicted]
                        print(specific_makeup_description)
                        response_data = {
                        "average_colors_rgb": average_colors_dict,
                        "average_colors_hex" : {key: rgb_to_hex(value) for key, value in average_colors_dict.items()}, 
                        "average_colors_names": average_colors_names_dict,
                        "Blush" : blush_dict,
                        "Lipstick" : lipstick_dict,
                        "Eyeshadow" : eyeshadow_dict,
                        "Foundation" : foundation_dict,
                        "uploaded_image": temp_filename,
                        "kategori_makeup": f"<b>{label_predicted}</b>",
                        "kategori_makeup_en": f"<b>{label_predicted}</b>",
                        "deskripsi_makeup": specific_makeup_description.get("deskripsi_makeup", ""),
                        "deskripsi_makeup_en": specific_makeup_description.get("deskripsi_makeup_en", ""),
                        "rekomendasi_tema_makeup": recommended_themes,
                        "pemilihan_warna": specific_makeup_description.get("pemilihan_warna", ""),
                        "pemilihan_warna_en": specific_makeup_description.get("pemilihan_warna_en", ""),
                        "tips_makeup_lain": specific_makeup_description.get("tips_makeup_lain", ""),
                        "tips_makeup_lain_en": specific_makeup_description.get("tips_makeup_lain_en", ""),
                        "kecocokan_makeup": True,
                        "rekomendasi_makeup_eyeshadow" : rekomendasi_makeup_eyeshadows,
                        "rekomendasi_makeup_blushon" : rekomendasi_makeup_blushons,
                        "rekomendasi_makeup_lipstick" : rekomendasi_makeup_lipsticks,
                        "rekomendasi_makeup_overall" : rekomendasi_makeup_overalls
                        }

                    else:
                        if label_predicted == "no_makeup":
                            recommended_themes = []
                            for makeup_label, makeup_info in makeup_data.items():
                                makeup_label = makeup_label.replace(" ", "_")
                                if tujuan_makeup in makeup_info["tujuan_makeup"]:
                                    recommended_themes.append(makeup_label)

                        recommended_themes = []
                        for makeup_label, makeup_info in makeup_data.items():
                            makeup_label = makeup_label.replace(" ", "_")
                            if tujuan_makeup in makeup_info["tujuan_makeup"]:
                                recommended_themes.append(makeup_label)

                        if "no_makeup" in recommended_themes:
                            recommended_themes.remove("no_makeup")
                        
                        rekomendasi_makeup_eyeshadows = []
                        rekomendasi_makeup_blushons = []
                        rekomendasi_makeup_lipsticks = []
                        rekomendasi_makeup_overalls = []
                        for recommended_theme in recommended_themes:
                            if recommended_theme == 'vintage makeup':
                                for i in range(1, 7):  # Loop dari 1 hingga 3
                                    rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                                for i in range(1, 4):
                                    rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{recommended_theme}{i}.png"
                                    rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                                    rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                                    rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)
                            
                            elif recommended_theme == 'fantasy makeup':
                                for i in range(1, 5):
                                    rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                                for i in range(1, 4):
                                    rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{recommended_theme}{i}.png"
                                    rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                                    rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                                    rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)

                            else:
                                for i in range(1, 4):  # Loop dari 1 hingga 4
                                    rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{recommended_theme}{i}.jpg"
                                    rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{recommended_theme}{i}.png"
                                    rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                                    rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                                    rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                                    rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)

                        tujuan_makeup_en_dict = {
                                "Sekolah": "School",
                                "Kerja": "Work",
                                "Kuliah": "College",
                                "Travelling": "Travelling",
                                "Halloween": "Halloween",
                                "Konser": "Concert",
                                "Cosplay": "Cosplay",
                                "Wedding": "Wedding",
                                "Graduation": "Graduation",
                                "Prom Night": "Prom Night",
                                "Party": "Party"
                            }
                        
                        if tujuan_makeup in tujuan_makeup_en_dict:
                            tujuan_makeup_en_str = tujuan_makeup_en_dict[tujuan_makeup]

                        if recommended_themes:
                            recommended_themes_str = ", ".join(recommended_themes).replace("_", " ")
                            response_data = {
                                "average_colors_rgb": average_colors_dict,
                                "average_colors_hex" : {key: rgb_to_hex(value) for key, value in average_colors_dict.items()}, 
                                "average_colors_names": average_colors_names_dict,
                                "Blush" : blush_dict,
                                "Lipstick" : lipstick_dict,
                                "Eyeshadow" : eyeshadow_dict,
                                "Foundation" : foundation_dict,
                                "uploaded_image": temp_filename,
                                "kategori_makeup": f"<b>{label_predicted}</b>",
                                "kategori_makeup_en": f"<b>{label_predicted}</b>",
                                "deskripsi_makeup": f"Tema make up kamu saat ini adalah <b>{label_predicted}</b>. Namun, jika tujuanmu adalah untuk <b>{tujuan_makeup}</b>, sebaiknya kamu pertimbangkan tema make up seperti <b>{recommended_themes_str}</b>.",
                                "deskripsi_makeup_en": f"The current theme of your makeup is <b>{label_predicted}</b>. However, if your goal is <b>{tujuan_makeup_en_str}</b>, you might want to consider makeup themes like <b>{recommended_themes_str}</b>.",
                                "rekomendasi_tema_makeup": recommended_themes,
                                "pemilihan_warna": specific_makeup_description.get("pemilihan_warna_negasi", ""),
                                "pemilihan_warna_en": specific_makeup_description.get("pemilihan_warna_negasi_en", ""),
                                "tips_makeup_lain": specific_makeup_description.get("tips_makeup_lain", ""),
                                "tips_makeup_lain_en": specific_makeup_description.get("tips_makeup_lain_en", ""),
                                "kecocokan_makeup": False,
                                "rekomendasi_makeup_eyeshadow" : rekomendasi_makeup_eyeshadows,
                                "rekomendasi_makeup_blushon" : rekomendasi_makeup_blushons,
                                "rekomendasi_makeup_lipstick" : rekomendasi_makeup_lipsticks,
                                "rekomendasi_makeup_overall" : rekomendasi_makeup_overalls
                                }
                        else:
                            response_data = {
                                "average_colors_rgb": average_colors_dict,
                                "average_colors_hex" : {key: rgb_to_hex(value) for key, value in average_colors_dict.items()}, 
                                "average_colors_names": average_colors_names_dict,
                                "Blush" : blush_dict,
                                "Lipstick" : lipstick_dict,
                                "Eyeshadow" : eyeshadow_dict,
                                "Foundation" : foundation_dict,
                                "uploaded_image": temp_filename,
                                "kategori_makeup": f"<b>{label_predicted}</b>",
                                "kategori_makeup_en": f"<b>{label_predicted}</b>",
                                "deskripsi_makeup": f"Tema make up kamu saat ini adalah <b>{label_predicted}</b>. Sayangnya, kami tidak memiliki rekomendasi yang cocok untuk tujuan make up '<b>{tujuan_makeup}<b>'.",
                                "deskripsi_makeup_en": f"The current theme of your makeup is <b>{label_predicted}</b>. Unfortunately, we don't have any suitable recommendations for the makeup goal '{tujuan_makeup_en_str}'.",
                                "kecocokan_makeup": False,
                                "rekomendasi_tema_makeup": recommended_themes,
                                "rekomendasi_makeup_eyeshadow" : rekomendasi_makeup_eyeshadows,
                                "rekomendasi_makeup_blushon" : rekomendasi_makeup_blushons,
                                "rekomendasi_makeup_lipstick" : rekomendasi_makeup_lipsticks,
                                "rekomendasi_makeup_overall" : rekomendasi_makeup_overalls
                            }
                    
                    with open('./result/hasil.json', 'w') as json_file:
                        json.dump(response_data, json_file, indent=4)
            
                    return JSONResponse(content=response_data)
        
        else:
            return JSONResponse("Wajah anda belum terdeteksi dengan baik oleh kamera / Your face has not been detected well by the camera")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/facialdetection_v2/")
async def facialdetection_v2(file: UploadFile, tujuan_makeup: TujuanMakeup2 = Form(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        if not allowed_photo(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")

        temp_filename_base = os.path.join(UPLOAD_FOLDER, "temp_image.jpg")

        with open(temp_filename_base, "wb") as temp_file:
            temp_file.write(file.file.read())

        temp_filename = cascade_mantap(temp_filename_base)

        if temp_filename != 0:
            img_raw = cv2.imread(temp_filename_base)
            detector = FaceLandMarks(staticMode=True)
            img, faces = detector.findFaceLandmark(img_raw)

            if len(faces) != 0:
                print("Number of faces:", len(faces))

                # List landmark yang ingin diambil warna rata-ratanya
                pipi_landmarks = [50, 330, 205, 101]
                bibir_landmarks = [15, 16, 72, 11, 85, 180]
                eye_shadow_landmarks = [222, 223, 224, 225, 442, 443, 444, 445]
                foundation_landmarks = [5, 195, 197, 248, 3]

                # Convert NumPy arrays to Python lists
                pipi_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in pipi_landmarks])]
                bibir_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in bibir_landmarks])]
                eye_shadow_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in eye_shadow_landmarks])]
                foundation_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in foundation_landmarks])]

                # Calculate average color for each makeup component
                average_blush_color = np.mean(pipi_colors, axis=0).astype(int).tolist()
                average_lipstick_color = np.mean(bibir_colors, axis=0).astype(int).tolist()
                average_eyeshadow_color = np.mean(eye_shadow_colors, axis=0).astype(int).tolist()
                average_foundation_color = np.mean(foundation_colors, axis=0).astype(int).tolist()

                # Create a dictionary to store the average colors
                average_colors_dict = {
                    "Blush": average_blush_color,
                    "Lipstick": average_lipstick_color,
                    "Eyeshadow": average_eyeshadow_color,
                    "Foundation": average_foundation_color
                }
                
                # Get color names for each average color
                average_colors_names_dict = {}
                for key, color in average_colors_dict.items():
                    color_rgb = tuple(color)
                    print(color_rgb)
                    color_hex = rgb_to_hex(color_rgb)
                    print(color_hex)
                    color_name = find_nearest_color_name(color_hex)
                    average_colors_names_dict[key] = color_name

                blush_dict = {
                    "average_colors_rgb": average_colors_dict['Blush'],
                    "average_colors_hex": rgb_to_hex(average_colors_dict["Blush"]),
                    "average_colors_names": average_colors_names_dict['Blush']
                }

                print(blush_dict)

                lipstick_dict = {
                    "average_colors_rgb": average_colors_dict['Lipstick'],
                    "average_colors_hex": rgb_to_hex(average_colors_dict["Lipstick"]),
                    "average_colors_names": average_colors_names_dict['Lipstick']
                }

                print(lipstick_dict)

                eyeshadow_dict = {
                    "average_colors_rgb": average_colors_dict['Eyeshadow'],
                    "average_colors_hex": rgb_to_hex(average_colors_dict["Eyeshadow"]),
                    "average_colors_names": average_colors_names_dict['Eyeshadow']
                }

                print(eyeshadow_dict)

                foundation_dict = {
                    "average_colors_rgb": average_colors_dict['Foundation'],
                    "average_colors_hex": rgb_to_hex(average_colors_dict["Foundation"]),
                    "average_colors_names": average_colors_names_dict['Foundation']
                }

                print(foundation_dict)

                # Ubah objek Image menjadi byte array (PNG)
                image_data_predicted = classify_makeup_colors(temp_filename_base)
                image_data_predicted_true = str(image_data_predicted[0])
                label_predicted = str(image_data_predicted[0]).replace("_", " ")
                print(label_predicted)

                # Load data from makeup_description.json
                with open('assets/json/makeup_description_v2.json', 'r') as makeup_description:
                    makeup_data = json.load(makeup_description)

                # Get makeup description based on predicted label
                specific_makeup_description = makeup_data.get(label_predicted)
                print(specific_makeup_description)

                rekomendasi_makeup_eyeshadows = []
                rekomendasi_makeup_blushons = []
                rekomendasi_makeup_lipsticks = []
                rekomendasi_makeup_overalls = []

                recommended_themes = []
                for makeup_label, makeup_info in makeup_data.items():
                    makeup_label = makeup_label.replace(" ", "_")
                    if tujuan_makeup in makeup_info["tujuan_makeup"]:
                        recommended_themes.append(makeup_label)

                if "no_makeup" in recommended_themes:
                    recommended_themes.remove("no_makeup")

                if label_predicted == 'vintage makeup':
                    for i in range(1, 7):  # Loop dari 1 hingga 6
                        rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                    for i in range(1, 4):
                        rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{image_data_predicted_true}{i}.png"
                        rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                        rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                        rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)
                    
                elif label_predicted == 'fantasy makeup':
                    for i in range(1, 5):
                        rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                    for i in range(1, 4):
                        rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{image_data_predicted_true}{i}.png"
                        rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                        rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                        rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)

                elif label_predicted == 'no makeup':
                    for i in range(1, 4):
                        rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_casual_makeup{i}.jpg"
                        rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_casual_makeup{i}.jpg"
                        rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_casual_makeup{i}.jpg"
                        rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_casual_makeup{i}.png"
                        rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                        rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                        rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)
                        rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)

                else:
                    for i in range(1, 4):  # Loop dari 1 hingga 4
                        rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{image_data_predicted_true}{i}.jpg"
                        rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{image_data_predicted_true}{i}.png"
                        rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                        rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                        rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                        rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)

                # Get the specific makeup description based on tujuan_makeup
                if label_predicted in makeup_data and tujuan_makeup in makeup_data[label_predicted]["tujuan_makeup"]:
                    specific_makeup_description = makeup_data[label_predicted]
                    print(specific_makeup_description)
                    response_data = {
                    "average_colors_rgb": average_colors_dict,
                    "average_colors_hex" : {key: rgb_to_hex(value) for key, value in average_colors_dict.items()}, 
                    "average_colors_names": average_colors_names_dict,
                    "Blush" : blush_dict,
                    "Lipstick" : lipstick_dict,
                    "Eyeshadow" : eyeshadow_dict,
                    "Foundation" : foundation_dict,
                    "uploaded_image": temp_filename,
                    "kategori_makeup": f"<b>{label_predicted}</b>",
                    "kategori_makeup_en": f"<b>{label_predicted}</b>",
                    "deskripsi_makeup": specific_makeup_description.get("deskripsi_makeup", ""),
                    "deskripsi_makeup_en": specific_makeup_description.get("deskripsi_makeup_en", ""),
                    "rekomendasi_tema_makeup": recommended_themes,
                    "pemilihan_warna": specific_makeup_description.get("pemilihan_warna", ""),
                    "pemilihan_warna_en": specific_makeup_description.get("pemilihan_warna_en", ""),
                    "tips_makeup_lain": specific_makeup_description.get("tips_makeup_lain", ""),
                    "tips_makeup_lain_en": specific_makeup_description.get("tips_makeup_lain_en", ""),
                    "kecocokan_makeup": True,
                    "rekomendasi_makeup_eyeshadow" : rekomendasi_makeup_eyeshadows,
                    "rekomendasi_makeup_blushon" : rekomendasi_makeup_blushons,
                    "rekomendasi_makeup_lipstick" : rekomendasi_makeup_lipsticks,
                    "rekomendasi_makeup_overall" : rekomendasi_makeup_overalls
                    }

                else:
                    if label_predicted == "no_makeup":
                        recommended_themes = []
                        for makeup_label, makeup_info in makeup_data.items():
                            makeup_label = makeup_label.replace(" ", "_")
                            if tujuan_makeup in makeup_info["tujuan_makeup"]:
                                recommended_themes.append(makeup_label)

                    recommended_themes = []
                    for makeup_label, makeup_info in makeup_data.items():
                        makeup_label = makeup_label.replace(" ", "_")
                        if tujuan_makeup in makeup_info["tujuan_makeup"]:
                            recommended_themes.append(makeup_label)

                    if "no_makeup" in recommended_themes:
                        recommended_themes.remove("no_makeup")
                    
                    rekomendasi_makeup_eyeshadows = []
                    rekomendasi_makeup_blushons = []
                    rekomendasi_makeup_lipsticks = []
                    rekomendasi_makeup_overalls = []
                    for recommended_theme in recommended_themes:
                        if recommended_theme == 'vintage makeup':
                            for i in range(1, 7):  # Loop dari 1 hingga 3
                                rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                            for i in range(1, 4):
                                rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{recommended_theme}{i}.png"
                                rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                                rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                                rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)
                        
                        elif recommended_theme == 'fantasy makeup':
                            for i in range(1, 5):
                                rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                            for i in range(1, 4):
                                rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{recommended_theme}{i}.png"
                                rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                                rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                                rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)

                        else:
                            for i in range(1, 4):  # Loop dari 1 hingga 4
                                rekomendasi_makeup_eyeshadow = f"./rekomendasi_makeup/eyeshadow_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_blushon = f"./rekomendasi_makeup/blushon_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_lipstick = f"./rekomendasi_makeup/lipstick_{recommended_theme}{i}.jpg"
                                rekomendasi_makeup_overall = f"./rekomendasi_makeup/overall_{recommended_theme}{i}.png"
                                rekomendasi_makeup_overalls.append(rekomendasi_makeup_overall)
                                rekomendasi_makeup_eyeshadows.append(rekomendasi_makeup_eyeshadow)
                                rekomendasi_makeup_blushons.append(rekomendasi_makeup_blushon)
                                rekomendasi_makeup_lipsticks.append(rekomendasi_makeup_lipstick)

                    tujuan_makeup_en_dict = {
                            "Sekolah/Kuliah": "School/College",
                            "Kerja": "Work",
                            "Travelling": "Travelling",
                            "Halloween": "Halloween",
                            "Konser": "Concert",
                            "Cosplay": "Cosplay",
                            "Wedding": "Wedding",
                            "Graduation": "Graduation",
                            "Prom Night": "Prom Night",
                            "Party": "Party"
                        }
                    
                    if tujuan_makeup in tujuan_makeup_en_dict:
                        tujuan_makeup_en_str = tujuan_makeup_en_dict[tujuan_makeup]

                    if recommended_themes:
                        recommended_themes_str = ", ".join(recommended_themes).replace("_", " ")
                        response_data = {
                            "average_colors_rgb": average_colors_dict,
                            "average_colors_hex" : {key: rgb_to_hex(value) for key, value in average_colors_dict.items()}, 
                            "average_colors_names": average_colors_names_dict,
                            "Blush" : blush_dict,
                            "Lipstick" : lipstick_dict,
                            "Eyeshadow" : eyeshadow_dict,
                            "Foundation" : foundation_dict,
                            "uploaded_image": temp_filename,
                            "kategori_makeup": f"<b>{label_predicted}</b>",
                            "kategori_makeup_en": f"<b>{label_predicted}</b>",
                            "deskripsi_makeup": f"Tema make up kamu saat ini adalah <b>{label_predicted}</b>. Namun, jika tujuanmu adalah untuk <b>{tujuan_makeup}</b>, sebaiknya kamu pertimbangkan tema make up seperti <b>{recommended_themes_str}</b>.",
                            "deskripsi_makeup_en": f"The current theme of your makeup is <b>{label_predicted}</b>. However, if your goal is <b>{tujuan_makeup_en_str}</b>, you might want to consider makeup themes like <b>{recommended_themes_str}</b>.",
                            "rekomendasi_tema_makeup": recommended_themes,
                            "pemilihan_warna": specific_makeup_description.get("pemilihan_warna_negasi", ""),
                            "pemilihan_warna_en": specific_makeup_description.get("pemilihan_warna_negasi_en", ""),
                            "tips_makeup_lain": specific_makeup_description.get("tips_makeup_lain", ""),
                            "tips_makeup_lain_en": specific_makeup_description.get("tips_makeup_lain_en", ""),
                            "kecocokan_makeup": False,
                            "rekomendasi_makeup_eyeshadow" : rekomendasi_makeup_eyeshadows,
                            "rekomendasi_makeup_blushon" : rekomendasi_makeup_blushons,
                            "rekomendasi_makeup_lipstick" : rekomendasi_makeup_lipsticks,
                            "rekomendasi_makeup_overall" : rekomendasi_makeup_overalls
                            }
                    else:
                        response_data = {
                            "average_colors_rgb": average_colors_dict,
                            "average_colors_hex" : {key: rgb_to_hex(value) for key, value in average_colors_dict.items()}, 
                            "average_colors_names": average_colors_names_dict,
                            "Blush" : blush_dict,
                            "Lipstick" : lipstick_dict,
                            "Eyeshadow" : eyeshadow_dict,
                            "Foundation" : foundation_dict,
                            "uploaded_image": temp_filename,
                            "kategori_makeup": f"<b>{label_predicted}</b>",
                            "kategori_makeup_en": f"<b>{label_predicted}</b>",
                            "deskripsi_makeup": f"Tema make up kamu saat ini adalah <b>{label_predicted}</b>. Sayangnya, kami tidak memiliki rekomendasi yang cocok untuk tujuan make up '<b>{tujuan_makeup}<b>'.",
                            "deskripsi_makeup_en": f"The current theme of your makeup is <b>{label_predicted}</b>. Unfortunately, we don't have any suitable recommendations for the makeup goal '{tujuan_makeup_en_str}'.",
                            "kecocokan_makeup": False,
                            "rekomendasi_tema_makeup": recommended_themes,
                            "rekomendasi_makeup_eyeshadow" : rekomendasi_makeup_eyeshadows,
                            "rekomendasi_makeup_blushon" : rekomendasi_makeup_blushons,
                            "rekomendasi_makeup_lipstick" : rekomendasi_makeup_lipsticks,
                            "rekomendasi_makeup_overall" : rekomendasi_makeup_overalls
                        }
                
                return JSONResponse(content=response_data)
        else:
            return JSONResponse("Wajah anda belum terdeteksi dengan baik oleh kamera / Your face has not been detected well by the camera")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)