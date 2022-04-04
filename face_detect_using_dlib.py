import dlib
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
shape_pred = dlib.shape_predictor("./dlib_face_reco_files/shape_predictor_5_face_landmarks.dat")
image = dlib.load_rgb_image("./find_name_images/python.jpg")
face_detected = detector(image,1)
for item in range(len(face_detected)):
    image_shape = shape_pred(image,face_detected[item])
    image_align = dlib.get_face_chip(image,image_shape)
    plt.imshow(image_align)
    plt.show()