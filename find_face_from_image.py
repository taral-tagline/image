import cv2

# defining size of images
(width, height) = (100,100)

haar_file = './haarcascade_xml_files/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
img = cv2.imread("./Images/python.jpg",0)

faces = face_cascade.detectMultiScale(img,1.31,minNeighbors=4)
image = 0

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = img[y:y + h, x:x + w]
    face_resize = cv2.resize(face, (width,height))
    name = './face_from_image/' + str(image) + '.jpg'
    cv2.imshow("Frame", face)
    cv2.waitKey(0)
    # writing the extracted images
    cv2.imwrite(name, face_resize)  
    # increasing counter so that it will
    # show how many frames are created
    image += 1
