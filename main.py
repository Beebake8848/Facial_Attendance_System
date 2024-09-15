import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendance-6d369-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "faceattendance-6d369.appspot.com"
})

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the encoding file
#print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
#print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
bucket = storage.bucket()
imgStudent = []

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print("matches", matches)
        #print("faceDis", faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            #print("Known Face Detected")
            #print(studentIds[matchIndex])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
            id = studentIds[matchIndex]

            if counter == 0:
                counter = 1
                modeType = 1

    if counter != 0:
        if counter == 1:
            studentInfo = db.reference(f'Students/{id}').get()
            #print(studentInfo)

            blob = bucket.get_blob(f'Images/{id}.jpeg')
            array = np.frombuffer(blob.download_as_string(), np.uint8)
            imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

            datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                               "%Y-%m-%d %H:%M:%S")
            secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
            #print(secondsElapsed)
            if secondsElapsed > 3:
                ref = db.reference(f'Students/{id}')
                studentInfo['total_attendance'] += 1
                ref.child('total_attendance').set(studentInfo['total_attendance'])
                ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                modeType = 3
                counter = 0
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        # Determine the dimensions of the region in imgBackground where imgStudent will be assigned
        region_height, region_width, _ = imgBackground[175:175 + 216, 909:909 + 216].shape

        # Resize imgStudent to fit within the region in imgBackground
        resized_imgStudent = cv2.resize(imgStudent, (region_width, region_height))

        # Assign the resized_imgStudent to the region in imgBackground
        imgBackground[175:175 + region_height, 909:909 + region_width] = resized_imgStudent

        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(id), (1006, 493),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(studentInfo['year']), (910, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(imgBackground, str(studentInfo['gender']), (1025, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        offset = (414 - w) // 2
        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

        counter += 1

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
