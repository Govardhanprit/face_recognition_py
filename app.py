import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, request, redirect, url_for

app = Flask(__name__)
known_face_encodings = []
known_face_names = []
known_faces_dir = '/Users/govardhanprit/Documents/Tst/my_app/known_faces' 
capture = None
stop_video = False


def load_known_faces():
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])

load_known_faces()
process_this_frame = True

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global capture
    if capture is None: 
        capture = cv2.VideoCapture(0)
    
   
    while True:
        if stop_video:
            break
        success, frame = capture.read()
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1]) 

            face_locations = []
            face_encodings = []
            face_names = []
            global process_this_frame

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if face_locations:  # Check if any faces were found
                    try:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    except Exception as e:
                        print(f"Error in face_encodings: {e}")
                        continue  # Skip this frame if there's an error

                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                        face_names.append(name)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['GET', 'POST'])
def train():
    global capture
    if request.method == 'POST':
        name = request.form['name']
        success, frame = capture.read()
        if success and name:
            file_path = os.path.join(known_faces_dir, f'{name}.jpg')
            cv2.imwrite(file_path, frame)
            #capture.release()
            load_known_faces()
            return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')
@app.route('/stop')
def stop():
    global stop_video
    stop_video = True
    os._exit(0)  # Forcefully exit the program
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
