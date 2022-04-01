import sys,os,numpy,cv2

haar_file = 'haarcascade_frontalface_default.xml'

# All the faces data will be store in a this folder
datasets = 'datasets'

# These are sub data sets of folder,
# for my faces I've used my name
sub_data = 'taral'

path = os.path.join(datasets,sub_data)
#print(path)
#os.mkdir(path)
if not os.path.isdir(path):
    os.mkdir(path)

# defining size of images
(width, height) = (130,100)

# store data for capture the image of video or webcam if it is avbilable
face_cascade = cv2.CascadeClassifier(haar_file)
video = cv2.VideoCapture("face_video.mp4")

# The program loops until it has 30 images of the face.
count = 1
while count< 100 :
    (_,im) = video.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width,height))
        cv2.imshow('face_resize',face_resize)
        #cv2.imwrite('%s s/% s.png' % (path,count),face_resize)
        #name = './datasets/taral' + str(count) + '.png'
        #cv2.imwrite(name, face_resize)
    count += 1
    
    cv2.imshow('OpenCV',im)
    key = cv2.waitKey(50)
    if key == 27:
        break