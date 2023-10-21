import numpy as np
import cv2
import os
from flask import Flask, request, render_template, jsonify, send_file

from assets.cascade import cascade_mantap

from flask import Flask, request, jsonify, render_template
from webcolors import rgb_to_hex
from assets.facial_landmarks import (
    FaceLandMarks,
    extract_color_roi,
    find_nearest_color_name,
    classify_makeup_colors,
    classify_glasses_or_no
)

import cv2
import json
import numpy as np

app = Flask(__name__, template_folder='templates')

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 255, 0)

app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}
app.config['UPLOAD_FOLDER'] = './user_photo/'

def allowed_photo(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route("/", methods=['GET'])
# def home():
#     return render_template('home.html')
    
@app.route('/facialdetection', methods=['POST'])
def facialdetection():
    if request.method == 'POST':
        # Get the uploaded image from the form data
        file = request.files['file']
        tujuan_makeup = str(request.form.get("tujuan_makeup")) #opsi : Halloween, Konser, Cosplay, Wedding, Graduation, Prom Night, Party, Sekolah, Kerja, Kuliah, Travelling
        if file:
            storage_dir = "./user_photo/"
            os.makedirs(storage_dir, exist_ok=True)

            temp_filename_base = os.path.join(storage_dir, "temp_image.jpg")
            file.save(temp_filename_base)
            print(temp_filename_base)
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
                            # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

                    # Save the image with landmarks (optional)
                    cv2.imwrite("./user_photo/image_with_landmarks.jpg", img)

                    # Display the image with landmarks
                    # cv2.imshow("Facial Landmarks", img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # List landmark yang ingin diambil warna rata-ratanya
                    pipi_landmarks = [50, 330, 205, 101]
                    bibir_landmarks = [15, 16, 72, 11, 85, 180]
                    eye_shadow_landmarks = [222, 223, 224, 225, 442, 443, 444, 445]
                    foundation_landmarks = [5, 195, 197, 248, 3]

                    # for pipi_index in pipi_landmarks:
                    #     x, y = faces[0][pipi_index]
                        # cv2.rectangle(img, (x - 25, y - 25), (x + 25, y + 25), (0, 255, 0), 2)

                    # cv2.imwrite("foto.jpg", img)
                    # cv2.imshow("Facial Landmarks", img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    # Convert NumPy arrays to Python lists
                    pipi_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in pipi_landmarks])]
                    bibir_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in bibir_landmarks])]
                    eye_shadow_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in eye_shadow_landmarks])]
                    foundation_colors = [color.tolist() for color in extract_color_roi(img, [faces[0][i] for i in foundation_landmarks])]
                    # print("######################################")
                    # print(pipi_colors)

                    # Calculate average color for each makeup componenT

                    # print("######################################")
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
                        # label_predicted = label_predicted.replace("_", " ")
                        # label_predicted = re.sub(r"_makeup.*", "_makeup", label_predicted)
                        # result_image_predicted = "image0.jpg"

                        # Load data from makeup_description.json
                        with open('assets/json/makeup_description.json', 'r') as makeup_description:
                            makeup_data = json.load(makeup_description)

                        # Get makeup description based on predicted label
                        specific_makeup_description = makeup_data.get(label_predicted)
                        print(specific_makeup_description)

                        print(tujuan_makeup)
                        
                        rekomendasi_makeup_eyeshadows = []
                        rekomendasi_makeup_blushons = []
                        rekomendasi_makeup_lipsticks = []
                        rekomendasi_makeup_overalls = []
                        # print(rekomendasi_makeup_eyeshadows)

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

                        print(rekomendasi_makeup_eyeshadows)

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
                            # "kecocokan_makeup": result_image_predicted
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
                                    # "Kuliah": "College",
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

                        return jsonify(response_data)
                    
                    else:
                        return jsonify("Anda masih menggunakan kacamata. Harap lepas agar kami dapat melakukan deteksi dengan maksimal. / You are still wearing glasses. Please remove them so that we can perform the detection at its maximum capability.")
                else:
                    return jsonify("Wajah anda belum terdeteksi dengan baik oleh kamera / Your face has not been detected well by the camera")

    # return render_template("facialdetection.html")

@app.route('/rekomendasi_makeup/<filename>')
def get_rekomendasi_makeup(filename):
    rekomendasi_makeup_folder = os.path.join(app.root_path, 'rekomendasi_makeup')
    return send_file(os.path.join(rekomendasi_makeup_folder, filename))

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = app.root_path
    app.run(debug=True)
