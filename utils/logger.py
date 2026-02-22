import csv
from datetime import datetime

class Logger:

    def __init__(self):
        self.file = open("fatigue_log.csv", mode="w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Time", "EAR", "MAR", "Pitch", "Yaw", "Status"])

    def log(self, ear, mar, pitch, yaw, status):
        self.writer.writerow([
            datetime.now().strftime("%H:%M:%S"),
            round(ear, 3),
            round(mar, 3),
            round(pitch, 2),
            round(yaw, 2),
            status
        ])