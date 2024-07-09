import cvlib as cv
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import json
import os
import requests

def download_file(url, dest_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(response.content)

def load_models():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    return faceNet, ageNet, genderNet

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def detect_gender_and_age(faceNet, ageNet, genderNet, frame):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        return resultImg, None, None

    padding = 20
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        return resultImg, gender, age

    return resultImg, None, None

class SignUpApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sign Up")
        self.geometry("500x500")
        
        self.create_widgets()
        self.verified = False
        
        # Ensure model files are downloaded
        self.ensure_model_files()

        # Load models
        self.faceNet, self.ageNet, self.genderNet = load_models()

    def create_widgets(self):
        tk.Label(self, text="Google Account:").place(x=50, y=50)
        self.google_account = tk.Entry(self)
        self.google_account.place(x=200, y=50)

        tk.Label(self, text="Password:").place(x=50, y=100)
        self.password = tk.Entry(self, show='*')
        self.password.place(x=200, y=100)

        tk.Label(self, text="Confirm Password:").place(x=50, y=150)
        self.confirm_password = tk.Entry(self, show='*')
        self.confirm_password.place(x=200, y=150)

        tk.Label(self, text="Name:").place(x=50, y=200)
        self.name = tk.Entry(self)
        self.name.place(x=200, y=200)

        tk.Label(self, text="Age:").place(x=50, y=250)
        self.age = tk.Entry(self)
        self.age.place(x=200, y=250)

        tk.Label(self, text="Phone Number:").place(x=50, y=300)
        self.phone_number = tk.Entry(self)
        self.phone_number.place(x=200, y=300)

        tk.Label(self, text="Emergency Contact Number:").place(x=50, y=350)
        self.emergency_contact = tk.Entry(self)
        self.emergency_contact.place(x=200, y=350)

        self.signup_btn = tk.Button(self, text="Sign Up", command=self.signup, state=tk.DISABLED)
        self.signup_btn.place(x=220, y=400)

        self.verify_btn = tk.Button(self, text="Verify with Webcam", command=self.run_webcam)
        self.verify_btn.place(x=190, y=450)

    def ensure_model_files(self):
        model_dir = os.path.expanduser("~/.cvlib/pre-trained")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        proto_path = os.path.join(model_dir, "gender_deploy.prototxt")
        model_path = os.path.join(model_dir, "gender_net.caffemodel")

        if not os.path.exists(proto_path):
            print("Downloading gender_deploy.prototxt...")
            download_file("https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt", proto_path)
        
        if not os.path.exists(model_path):
            print("Downloading gender_net.caffemodel...")
            download_file("https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel", model_path)

    def run_webcam(self):
        webcam = cv2.VideoCapture(0)
        verified = False

        while webcam.isOpened():
            status, frame = webcam.read()
            if not status:
                break
            
            resultImg, gender, age = detect_gender_and_age(self.faceNet, self.ageNet, self.genderNet, frame)

            if gender == 'Male':
                verified = True

            cv2.imshow("Real-time gender detection", resultImg)

            if verified:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()

        if verified:
            messagebox.showinfo("Verification", "Verification complete! You may proceed.")
            self.verified = True
            self.signup_btn.config(state=tk.NORMAL)
        else:
            messagebox.showwarning("Verification", "Verification failed. Please try again.")
            self.verified = False
            self.signup_btn.config(state=tk.DISABLED)

    def signup(self):
        if not self.verified:
            messagebox.showwarning("Verification", "Please verify using the webcam first.")
            return

        data = {
            "google_account": self.google_account.get(),
            "password": self.password.get(),
            "confirm_password": self.confirm_password.get(),
            "name": self.name.get(),
            "age": self.age.get(),
            "phone_number": self.phone_number.get(),
            "emergency_contact": self.emergency_contact.get()
        }

        if data["password"] != data["confirm_password"]:
            messagebox.showwarning("Error", "Passwords do not match.")
            return

        with open("signup_data.json", "w") as f:
            json.dump(data, f)

        print("Authentication complete")
        messagebox.showinfo("Success", "Sign-up successful!")

if __name__ == "__main__":
    app = SignUpApp()
    app.mainloop()
