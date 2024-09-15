import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendance-6d369-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "faceattendance-6d369.appspot.com"
})

# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []

for path in pathList:
    if path.endswith('.jpeg') or path.endswith('.jpg') or path.endswith('.png'):
        try:
            img = cv2.imread(os.path.join(folderPath, path))
            if img is not None:
                imgList.append(img)
                studentIds.append(os.path.splitext(path)[0])
            else:
                print(f"Unable to read image file: {path}")
        except Exception as e:
            print(f"Error loading image file {path}: {e}")
    else:
        print(f"Skipping file: {path}")

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print(studentIds)

undetected_images = []

def findEncodings(imagesList):
    encodeList = []
    for img, path in zip(imagesList, pathList):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find face locations in the image
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) > 0:
            # If faces are found, encode the first face
            encode = face_recognition.face_encodings(img, face_locations)[0]
            encodeList.append(encode)
        else:
            print(f"No face found in the image: {path}")
            undetected_images.append(path)  # Add the filename to the list of undetected images
            # Append None to indicate no face found
            encodeList.append(None)
    return encodeList



print("Undetected Images:", undetected_images)

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnown = [enc for enc in encodeListKnown if enc is not None]  # Remove None values
print("Encoding Complete")

encodeListKnownWithIds = [encodeListKnown, studentIds]

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")
