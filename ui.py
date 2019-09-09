import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk

window = tk.Tk()

window.title("Leaf Disease Diagnose")

window.geometry("500x500")
window.configure(background ="lightskyblue")

title = tk.Label(text="Click below to choose picture for testing disease....", background = "lightskyblue", fg="white", font=("", 20))
title.grid()


def analysis():
    import cv2
    import numpy as np
    import os
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    validation_dir = 'test'
    IMG_SIZE = 72
    LR = 1e-3
    MODEL_NAME = 'leafDectection-{}.model'.format(LR)

    def process_validate_data():
        validation_data = []
        for img in tqdm(os.listdir(validation_dir)):
            path = os.path.join(validation_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            validation_data.append([np.array(img), img_num])
        np.save('validation_data.npy', validation_data)
        return validation_data

    validation_data = process_validate_data()
    #validation_data = np.load('validation_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.reset_default_graph()

    network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3])
# IMG_SIZE = 72, LR = 1e-3
    network = conv_2d(network, 64, 3, strides=3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)

    network = conv_2d(network, 128, 3, strides=3, activation='relu')
    network = max_pool_2d(network, 3, strides = 2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)

    network = conv_2d(network, 128, 1, activation='relu')

    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 64, 3, activation='relu')
    network = avg_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 5, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=LR)

    model = tflearn.DNN(network, tensorboard_dir='logs')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(validation_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'bacterial_blight'
        elif np.argmax(model_out) == 2:
            str_label = 'cercospora_leaf_spot'
        elif np.argmax(model_out) == 3:
            str_label = 'alternaria_alternata'
        elif np.argmax(model_out) == 4:
            str_label = 'anthracnose'

        if str_label =='healthy':
            status ="HEALTHY"
        else:
            status = "UNHEALTHY"

        message = tk.Label(text='Status: ' + status, background="white", fg="Brown", font=("", 15))
        message.grid(column=0, row=3, padx=10, pady=10)

        if str_label == 'bacterial_blight':
            diseasename = "Bacterial Blight "
            disease = tk.Label(text='Disease Name: ' + diseasename, background="white", fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
        elif str_label == 'cercospora_leaf_spot':
            diseasename = "Cercospoara Leaf Spot "
            disease = tk.Label(text='Disease Name: ' + diseasename, background="white", fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
        elif str_label == 'alternaria_alternata':
            diseasename = "Alternaria Alternata"
            disease = tk.Label(text='Disease Name: ' + diseasename, background="white", fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
        elif str_label == 'anthracnose':
            diseasename = "Anthracnose"
            disease = tk.Label(text='Disease Name: ' + diseasename, background = "white", fg = "Black", font=("", 15))
            disease.grid(column = 0, row = 4, padx = 10, pady = 10)
        else:
            r = tk.Label(text='Plant is healthy', background="white", fg="Black", font=("", 15))
            r.grid(column=0, row=4, padx=10, pady=10)

        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)

def openphoto():
    dirPath = "test"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    fileName = askopenfilename(initialdir='/Users/chenlinyue/Desktop', title='Select image for analysis', filetypes=[('image files', '.jpg')])
    dst = "/Users/chenlinyue/Desktop/LeafDiseaseDetectionTest/test"
    shutil.copy(fileName, dst)
    load = Image.open(fileName)

    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="250")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=125, pady = 10)

    title.destroy()
    button1.destroy()

    button2 = tk.Button(text="Analyse Image", command=analysis, bg="lightskyblue")
    button2.grid(column=0, row=2, padx=10, pady=10)

button1 = tk.Button(text="Get Photo", command = openphoto, bg="lightskyblue")
button1.grid(column=0, row=2, padx=10, pady=10)


window.mainloop()
