import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
import os
import csv
from tkinter import messagebox

# model
model = tf.keras.models.Sequential()

# vgg16
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

# weight
model.load_weights('C:/Users/WINDOWS/PycharmProjects/ImageClassificationForMusicGenre/vgg16model/model.h5')

genre_labels = ['Blues', 'Country', 'EDM', 'Jazz', 'Metal', 'Pop', 'Rock']


# tk.root
root = tk.Tk()
root.title("Genre Prediction")

# open gambar dan display gambar
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        predict_image(file_path)

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        predict_folder(folder_path)

def predict_image(image_path):
    # open dan preprocess gambar
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    image_data = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=image_data)
    canvas.image = image_data  # buat canvas object

    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediksi
    prediction = model.predict(img_array)
    predicted_genre_index = np.argmax(prediction)
    predicted_genre = genre_labels[predicted_genre_index]
    confidence = prediction[0][predicted_genre_index] * 100

    # update bagian prediksi dan confidence
    prediction_label.config(text="Predicted Genre: " + predicted_genre)
    confidence_label.config(text="Confidence: {:.2f}%".format(confidence))

    # menampilkan confidence dan prediksi di bawah gambar
    prediction_label.place(x=0, y=320)
    confidence_label.place(x=0, y=340)

def predict_folder(folder_path):
    # list untuk menyimpan prediksi saat memprediksi folder
    predictions = []

    # for untuk prediksi setiap gambar di folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)

            # buka dan preprocess gambar
            img = Image.open(image_path)
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # prediksi
            prediction = model.predict(img_array)
            predicted_genre_index = np.argmax(prediction)
            predicted_genre = genre_labels[predicted_genre_index]
            confidence = prediction[0][predicted_genre_index] * 100

            # simpan prediksi
            predictions.append({
                'Image': filename,
                'Prediction': predicted_genre,
                'Confidence': confidence,
                'True Genre': os.path.basename(folder_path)
            })

    # simpan semua prediksi yang di store ke file tipe csv
    folder_name = os.path.basename(folder_path)
    csv_path = f'{folder_name}_predictions.csv'
    save_predictions_to_csv(predictions, csv_path)

def save_predictions_to_csv(predictions, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Image', 'Prediction', 'Confidence', 'True Genre'])
        writer.writeheader()
        writer.writerows(predictions)

    print(f'Predictions saved to: {csv_path}')
    messagebox.showinfo("CSV Complete!", "CSV file has been created successfully.")

# gui element seperti tombol dll
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

select_folder_button = tk.Button(root, text="Select Folder", command=select_folder)
select_folder_button.pack(pady=10)

canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

prediction_label = tk.Label(root, text="Predicted Genre:")
prediction_label.pack(pady=5)

confidence_label = tk.Label(root, text="Confidence:")
confidence_label.pack(pady=5)

# tkinter event loop
root.mainloop()