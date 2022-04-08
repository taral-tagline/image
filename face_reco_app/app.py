import cv2
import os
import face_recognition
import pickle
import numpy as np
from flask import Flask, render_template, request, Response
from flask_sqlalchemy import SQLAlchemy

UPLOAD_FOLDER = './upload_photos/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
knownEncodings = []
knownNames = []

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Upload(db.Model):
    id = db.Column(db.Integer,primary_key = True)
    name = db.Column(db.String(50))
    data = db.Column(db.LargeBinary)

video_capture = cv2.VideoCapture(0)
cascPathface = "haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascPathface)


def gen_frames():
    with open("face_enc", "rb") as f:
        data = pickle.load(f)
    # with open('face_enc','rb') as f :
    #     print(f.readlines())
    # data = pickle.loads(open('face_enc', "rb").read())
    # print(data)
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0),fx=0.25,fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(data["encodings"],face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(data["encodings"],face_encoding)
                best_metch_index = np.argmin(face_distances)
                if matches[best_metch_index]:
                    name = data["names"][best_metch_index]
                face_names.append(name)
        process_this_frame = not process_this_frame

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(face_locations, face_names) :
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/',methods = ['POST','GET'])  
def index():
    if request.method == 'POST':  
        uname=request.form['input_name']
        upload_file = request.files['input_file']
        filename = upload_file.filename
        # # Store Form Data into table  
        # upload = Upload(name=uname , data = upload_file.read())
        # db.session.add(upload)
        # db.session.commit()

        # Store image in folder
        upload_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        upload_image = cv2.imread(UPLOAD_FOLDER + upload_file.filename)
        rgb = cv2.cvtColor(upload_image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb,model='cnn')
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(uname)

        add_data = {"encodings": knownEncodings, "names": knownNames}
        f = open("face_enc", "ab")
        f.write(pickle.dumps(add_data))
        f.close()
        # print(add_data)
        simple_msg = f'File Uploaded name : {upload_file.filename} and your name is %s' %uname

        return render_template('index.html',simple_msg = simple_msg)
    else:
        return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug = True)