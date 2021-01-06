import cv2
import numpy as np
import face_recognition

imgRF = face_recognition.load_image_file('ImagesBasic/RF1.jpg')
imgRF = cv2.cvtColor(imgRF, cv2.COLOR_BGR2RGB)
imgRF_test = face_recognition.load_image_file('ImagesBasic/RF2.jpg')
imgRF_test = cv2.cvtColor(imgRF_test, cv2.COLOR_BGR2RGB)

face_Loc = face_recognition.face_locations(imgRF)[0]
encoded_RF = face_recognition.face_encodings(imgRF)[0]
cv2.rectangle(imgRF, (face_Loc[3], face_Loc[0]), (face_Loc[1], face_Loc[2]), (255, 0, 255), 2)

face_Loc_test = face_recognition.face_locations(imgRF_test)[0]
encoded_RF_test = face_recognition.face_encodings(imgRF_test)[0]
cv2.rectangle(imgRF_test, (face_Loc_test[3], face_Loc_test[0]), (face_Loc_test[1], face_Loc_test[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encoded_RF], encoded_RF_test)
face_dis = face_recognition.face_distance([encoded_RF], encoded_RF_test)
print(results, face_dis)
cv2.putText(imgRF_test, f'{results} {round(face_dis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Roger Federer', imgRF)
cv2.imshow('Roger Federer Test', imgRF_test)
cv2.waitKey(0)
