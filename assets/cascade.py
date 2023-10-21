import cv2

def cascade_mantap(temp_filename_base):
    # Inisialisasi detektor
    face_detector = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')
    nose_detector = cv2.CascadeClassifier('./haarcascade/haarcascade_mcs_nose.xml')
    mouth_detector = cv2.CascadeClassifier('./haarcascade/haarcascade_mcs_mouth.xml')
    eye_detector = cv2.CascadeClassifier('./haarcascade/haarcascade_eye.xml')

    img = cv2.imread(temp_filename_base)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    mouth = mouth_detector.detectMultiScale(gray, 1.5, 20)
    nose = nose_detector.detectMultiScale(gray, 1.5, 25)
    eyes = eye_detector.detectMultiScale(gray, 1.3, 15)

    print(len(faces))
    print(len(mouth))
    print(len(nose))
    print(len(eyes))

    # Periksa apakah semua bagian terdeteksi
    # if len(faces) > 0 and len(mouth) > 0 and len(nose) > 0 and len(eyes) > 0:
    #     return 1  # Semua bagian terdeteksi
    # else:
    #     return 0  # Salah satu atau lebih bagian tidak terdeteksi
    
    if len(faces) > 0 and len(mouth) > 0 and len(eyes) > 0:
        return 1  # Semua bagian terdeteksi
    else:
        return 0  # Salah satu atau lebih bagian tidak