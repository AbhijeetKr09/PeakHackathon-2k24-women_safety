import tkinter as tk
import threading
import time
import geocoder

class LocationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Location Printer")
        self.geometry("300x200")
        
        self.create_widgets()
        self.running = False
        self.thread = None

    def create_widgets(self):
        self.start_btn = tk.Button(self, text="Send Current Location (SOS)", command=self.start_printing)
        self.start_btn.pack(pady=20)
        
        self.stop_btn = tk.Button(self, text="STOP SENDING", command=self.stop_printing, state=tk.DISABLED)
        self.stop_btn.pack(pady=20)

    def start_printing(self):
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.thread = threading.Thread(target=self.print_location_loop)
        self.thread.start()

    def stop_printing(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.thread:
            self.thread.join()

    def print_location_loop(self):
        while self.running:
            try:
                g = geocoder.ip('me')
                if g.ok:
                    lat, lon = g.latlng
                    self.print_location(lat, lon)
                else:
                    print("Error getting location")
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(10)

    def print_location(self, lat, lon):
        print(f"Current Location - Latitude: {lat}, Longitude: {lon}")

if __name__ == "__main__":
    app = LocationApp()
    app.mainloop()
