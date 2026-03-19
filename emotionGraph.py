import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class EmotionGraph ():
    def __init__(self, master):
        # Emotion ranges (each band)
        self.master = master
        self.emotion_ranges = {
            "angry": (0),
            "sad": (100),
            "fear": (200),
            "disgust": (300),
            "neutral": (400),
            "happy": (500),
            "surprised": (600)
        }

        self.x_data = []
        self.y_data = []
        self.t = 0

        # This ensures a popup window instead of inline
        self.master.title("Emotion Flow")
        self.master.geometry("600x400")
        self.fig, self.ax = plt.subplots(figsize=(5, 2))

        # Updated limits for new ranges
        self.ax.set_ylim(0, 700)

        # Center labels in each band
        self.ax.set_yticks([50, 150, 250, 350, 450, 550, 650])
        self.ax.set_yticklabels([
            "angry", "sad", "fear", "disgust",
            "neutral", "happy", "surprised"
        ])

        # Draw band separators
        for y in [100, 200, 300, 400, 500, 600]:
            self.ax.axhline(y=y, linestyle='--', linewidth=0.5)

        self.ax.set_title("Emotion Flow")
        self.ax.set_xlabel("Time")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # 🔑 Convert emotion + score → y value
    def emotion_to_y(self, emotion, score):
        if emotion not in self.emotion_ranges:
            return None
        return self.emotion_ranges[emotion] + (score * 100)

    def update(self, emotion, score):

        y_val = self.emotion_to_y(emotion, score)

        if y_val is None:
            return

        self.x_data.append(self.t)
        self.y_data.append(y_val)
        self.t += 1

        self.ax.plot(self.x_data, self.y_data)

        # Keep last 30 points (scrolling effect)
        if len(self.x_data) > 30:
            self.x_data = self.x_data[-30:]
            self.y_data = self.y_data[-30:]

        self.canvas.draw()
        self.canvas.flush_events()
