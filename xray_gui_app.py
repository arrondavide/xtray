import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Load model
import os
model_path = os.path.join(os.path.dirname(__file__), "xray_cnn_model.h5")
model = load_model(model_path)

class_names = ["COVID", "PNEUMONIA", "NORMAL"]
img_size = 128

# Prediction function
def predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file_path:
        # Load and display image
        img = Image.open(file_path).convert('L')
        img_resized = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img_resized)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        try:
            # Prepare for prediction
            img = img.resize((img_size, img_size))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, img_size, img_size, 1)

            # Predict
            prediction = model.predict(img_array)
            result = class_names[np.argmax(prediction)]
            confidence = round(100 * np.max(prediction), 2)

            result_var.set(f"Prediction: {result} ({confidence}% confidence)")
        except Exception as e:
            result_var.set(f"❌ Error: {str(e)}")

try:
    root = tk.Tk()
    root.title("Chest X-ray Classifier")
    root.geometry("400x500")

    tk.Label(root, text="Chest X-ray Predictor", font=("Helvetica", 16)).pack(pady=10)
    tk.Button(root, text="Upload X-ray Image", command=predict).pack(pady=10)

    image_label = tk.Label(root)
    image_label.pack(pady=10)

    result_var = tk.StringVar()
    tk.Label(root, textvariable=result_var, font=("Helvetica", 14), fg="blue").pack(pady=20)

    tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

    root.mainloop()
except Exception as e:
    print("❌ GUI Error:", e)
    input("Press Enter to exit...")

