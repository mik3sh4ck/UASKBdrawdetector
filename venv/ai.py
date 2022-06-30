from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import os
import numpy as np

def load_dataset():
    kotak = []
    lingkaran = []
    segitiga = []
    garis = []
    orang = []

    for file in os.listdir("venv/kotak"):
        img = Image.open("venv/kotak/" + file)
        img = np.array(img)
        img = img.flatten()
        kotak.append(img)

    for file in os.listdir("venv/lingkaran"):
        img = Image.open("venv/lingkaran/" + file)
        img = np.array(img)
        img = img.flatten()
        lingkaran.append(img)

    for file in os.listdir("venv/segitiga"):
        img = Image.open("venv/segitiga/" + file)
        img = np.array(img)
        img = img.flatten()
        segitiga.append(img)

    for file in os.listdir("venv/garis"):
        img = Image.open("venv/garis/" + file)
        img = np.array(img)
        img = img.flatten()
        garis.append(img)
    
   

    return kotak, lingkaran, segitiga, garis, 

def load_ai():
    model = KNeighborsClassifier(n_neighbors=5)
    print("[INFO] Loading Dataset")
    kotak, lingkaran, segitiga, garis,  = load_dataset()
    print("[INFO] Loading Model")
    y_kotak = np.zeros(len(kotak))
    y_lingkaran = np.ones(len(lingkaran))
    y_segitiga = np.ones(len(segitiga)) * 2
    y_garis = np.ones(len(garis)) * 3
    X = kotak + lingkaran + segitiga + garis 
    y = np.concatenate([y_kotak, y_lingkaran, y_segitiga, y_garis ])
    model.fit(X, y)
    return model