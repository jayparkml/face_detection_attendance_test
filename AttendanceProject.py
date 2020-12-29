import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)


def find_encodings(images):
    encoded_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encoded_list.append(encoded)
    return encoded_list


def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            datetime_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datetime_string}')


encoded_list_known = find_encodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, cap_img = cap.read()
    img_small = cv2.resize(cap_img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(img_small)
    encodes_current_frame = face_recognition.face_encodings(img_small, faces_current_frame)

    for encoded_face, face_location in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encoded_list_known, encoded_face)
        face_distance = face_recognition.face_distance(encoded_list_known, encoded_face)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = classNames[match_index].upper()
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(cap_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(cap_img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(cap_img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow('Webcam', cap_img)
    key = cv2.waitKey(50)
    if key == ord('q'):
        break

# face_Loc = face_recognition.face_locations(imgRF)[0]
# encoded_RF = face_recognition.face_encodings(imgRF)[0]
# cv2.rectangle(imgRF, (face_Loc[3], face_Loc[0]), (face_Loc[1], face_Loc[2]), (255, 0, 255), 2)
#
# face_Loc_test = face_recognition.face_locations(imgRF_test)[0]
# encoded_RF_test = face_recognition.face_encodings(imgRF_test)[0]
# cv2.rectangle(imgRF_test, (face_Loc_test[3], face_Loc_test[0]), (face_Loc_test[1], face_Loc_test[2]), (255, 0, 255), 2)
#
# results = face_recognition.compare_faces([encoded_RF], encoded_RF_test)
# face_dis = face_recognition.face_distance([encoded_RF], encoded_RF_test)
