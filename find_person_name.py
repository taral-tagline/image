import face_recognition
from imutils import paths
import pickle
import cv2
 
#find path of xml file containing haarcascade file
cascPathface = "./haarcascade_xml_files/haarcascade_frontalface_default.xml"

# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)

# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())

imagePaths = list(paths.list_images('find_name_images'))

for (number_of_images, imagePath) in enumerate(imagePaths):

    # Find path to the image you want to detect face and pass it here
    image = cv2.imread(imagePath)
    # image = cv2.imread("./find_name_images/abhishek.jpeg")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert image to Greyscale for haarcascade
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(100, 100))

    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []

    # loop over the facial embeddings incase
    for encoding in encodings:
        # Compare encodings with encodings in data["encodings"]
        # Matches contain array with boolean values and True for the embeddings it matches closely and False for rest
        matches = face_recognition.compare_faces(data["encodings"],encoding, tolerance=0.55)
        
        # set name = Unknown if no encoding matches
        name = "Unknown"
        
        # check to see if we have found a match
        if True in matches:
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for each recognized face face
            for i in matchedIdxs:
                # Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                # increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
                # set name which has highest count
                name = max(counts, key=counts.get)
    
            # update the list of names
            names.append(name)
            
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names) :
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Frame", image)
        cv2.waitKey(0)