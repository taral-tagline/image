import cv2

haar_file = 'haarcascade_frontalface_default.xml'

# store data for capture the image of video or webcam if it is avbilable
face_cascade = cv2.CascadeClassifier(haar_file)
video = cv2.VideoCapture("face_video.mp4")

# The program loops until it has 30 images of the face.
count = 1
while count< 30 :
    (_,im) = video.read()
    faces = face_cascade.detectMultiScale(im,1.3,4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
    count += 1
    
    cv2.imshow('OpenCV',im)
    key = cv2.waitKey(30)
    if key == 27:
        break