import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import os


model = load_model("model_animals_clothes.h5")
class_names = sorted(os.listdir("dataset/train"))

def predict_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((150, 150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)
    class_id = np.argmax(pred)
    return class_names[class_id]

def open_image():
    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
    if path:
        img = Image.open(path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        result = predict_image(path)
        label_result.config(text=f"Это: {result}")

root = tk.Tk()
root.title("Классификатор (Кошка/Собака/Одежда)")

btn = tk.Button(root, text="Загрузить фото", command=open_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

label_result = tk.Label(root, text="", font=("Arial", 20))
label_result.pack()

root.mainloop()