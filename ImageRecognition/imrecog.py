import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import PIL
from PIL import Image,ImageGrab,ImageDraw
import tensorflow as tf
import tkinter as tk
from tkinter import *


mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,28,28,1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

loss,accuracy = model.evaluate(x_test,y_test)
print(accuracy)
print(loss)


def testing():
    img=cv2.imread('image.png')
    img=cv2.resize(img,(28,28))[:,:,0]
    img=np.invert(np.array([img]))
    prediction=model.predict(img)
    return prediction


classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
width = 500
height = 500
center = height // 2
white = (255, 255, 255)
green = (0, 128, 0)


def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=40)
    draw.line([x1, y1, x2, y2], fill="black", width=40)


def model1():
    filename = "image.png"
    image1.save(filename)
    pred = testing()
    print(pred[0])
    print('argmax', np.argmax(pred), '\n')
    txt.insert(tk.INSERT,
               "Predicted Value: {}".format(np.argmax(pred)))


def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))
    txt.delete('1.0', END)


root = Tk()

root.resizable(0, 0)
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()


image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

txt = tk.Text(root, bd=3, exportselection=0, bg='WHITE', font='Helvetica',
              padx=10, pady=10, height=5, width=20)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)


btnModel = Button(text="Predict", command=model1)
btnClear = Button(text="clear", command=clear)
btnModel.pack()
btnClear.pack()
txt.pack()
root.title('Digit Recognizer')
root.mainloop()