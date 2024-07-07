import cvlib as cv
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import json

class SignUpApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sign Up")
        self.geometry("500x500")
        
        self.create_widgets()
        self.verified = False

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

    def run_webcam(self):
        webcam = cv2.VideoCapture(0)
        padding = 20
        verified = False

        while webcam.isOpened():
            status, frame = webcam.read()
            face, confidence = cv.detect_face(frame)

            for idx, f in enumerate(face):
                (startX, startY) = max(0, f[0] - padding), max(0, f[1] - padding)
                (endX, endY) = min(frame.shape[1] - 1, f[2] + padding), min(frame.shape[0] - 1, f[3] + padding)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                face_crop = np.copy(frame[startY:endY, startX:endX])
                (label, confidence) = cv.detect_gender(face_crop)
                idx = np.argmax(confidence)
                label = label[idx]
                label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if "female" in label:
                    verified = True
                    break

            cv2.imshow("Real-time gender detection", frame)

            if verified:
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
