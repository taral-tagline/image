import cv2

# defining size of images
(width, height) = (100,100)

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
img = cv2.imread("python.jpg",0)

faces = face_cascade.detectMultiScale(img,1.31,minNeighbors=4)
image = 0
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    face = img[y:y + h, x:x + w]
    face_resize = cv2.resize(face, (width,height))
    # if video is still left continue creating images
    name = './Images/image' + str(image) + '.jpg'
    print ('Creating...' + name)  
    # writing the extracted images
    cv2.imwrite(name, face_resize)  
    # increasing counter so that it will
    # show how many frames are created
    image += 1
cv2.imshow("image",img)
cv2.waitKey(0)