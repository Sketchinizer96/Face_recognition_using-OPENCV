import numpy as np
import cv2
import os

import faceRecognition as fr
print (fr)

test_img = cv2.imread(r'C:\Users\A707272\PycharmProjects\Face_Recognition\Test_image\test1.jpg')  #Give path to the image which you want to test


faces_detected, gray_img = fr.faceDetection(test_img)
print("face Detected: ",faces_detected)

# Training begins here

faces,faceID=fr.labels_for_training_data(r'C:\Users\A707272\PycharmProjects\Face_Recognition\Training_images')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'C:\Users\A707272\PycharmProjects\Face_Recognition\trainingData.yml')

name={0:"Aniket", 1:" Ramavati"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print ("Confidence :",confidence)
    print("label :",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows