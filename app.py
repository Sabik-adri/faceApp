from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('vendatta.png', cv2.IMREAD_UNCHANGED)
scale_factor = 1.2

def detect_faces():
    webcam = cv2.VideoCapture(0)
    while True:
        successful_frame_read, frame = webcam.read()
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
        for (x, y, w, h) in face_coordinates:
            resized_width = int(w * scale_factor)
            resized_height = int(h * scale_factor)
            face_image = cv2.resize(img, (resized_width, resized_height))
            alpha_channel = face_image[:, :, 3] / 255.0
            frame = frame.astype(float)
            x_start = max(0, x - int((resized_width - w) / 2))
            y_start = max(0, y - int((resized_height - h) / 2))
            x_end = x_start + resized_width
            y_end = y_start + resized_height
            if x_end > frame.shape[1]:
                x_end = frame.shape[1]
            if y_end > frame.shape[0]:
                y_end = frame.shape[0]
            frame_roi = frame[y_start:y_end, x_start:x_end, :]
            alpha_channel = cv2.resize(alpha_channel, (x_end - x_start, y_end - y_start))
            for c in range(3):
                frame_roi[:, :, c] = alpha_channel * face_image[:, :, c] + (1 - alpha_channel) * frame_roi[:, :, c]
            frame = frame.astype('uint8')
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
