import cv2

haar_file = './haarcascade_xml_files/haarcascade_frontalface_default.xml'

# defining size of images
(width, height) = (150,150)

# store data for capture the image of video or webcam if it is avbilable
face_cascade = cv2.CascadeClassifier(haar_file)
video = cv2.VideoCapture("./video/face_video.mp4")

# The program loops until it has 30 images of the face.
count = 1
while count< 30 :
    (_,im) = video.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width,height))
        name = './face_from_video/' + str(count) + '.png'
        cv2.imwrite(name, face_resize)
    count += 1