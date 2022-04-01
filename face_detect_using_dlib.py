import dlib
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
shape_pred = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
#model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
image = dlib.load_rgb_image("./find_name/python.jpg")
face_detected = detector(image,1)
#print(len(face_detected))
for item in range(len(face_detected)):
    #print(item)
    image_shape = shape_pred(image,face_detected[item])
    #print(type(image_shape))
    image_align = dlib.get_face_chip(image,image_shape)
    plt.imshow(image_align)
    plt.show()

# import face_recognition
# import cv2

# image = face_recognition.load_image_file("./find_name/taral1.jpeg")
# face_locations = face_recognition.face_locations(image)
# #print(face_locations)
# face_landmarks_list = face_recognition.face_landmarks(image)
# #print(face_landmarks_list)

# color = (0, 0, 255)
# fontScale = 1
# thickness = 1
# for l in face_landmarks_list:
#     for key in l.keys():
#         for item in l[key]:
#             #cv2.putText(image,".",item, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
#             cv2.circle(image, item, 1, (0,0,255), -1)
# cv2.imshow("image",image)
# cv2.waitKey(0)