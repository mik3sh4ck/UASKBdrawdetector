from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import os
import numpy as np

def load_dataset():
    kotak = []
    lingkaran = []
    segitiga = []
    garis = []
    segilima = []
    trapesium = []

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
    
    for file in os.listdir("venv/segilima"):
        img = Image.open("venv/segilima/" + file)
        img = np.array(img)
        img = img.flatten()
        segilima.append(img)
    
    for file in os.listdir("venv/trapesium"):
        img = Image.open("venv/trapesium/" + file)
        img = np.array(img)
        img = img.flatten()
        trapesium.append(img)
    
   

    return kotak, lingkaran, segitiga, garis, segilima, trapesium

def load_ai():
    model = KNeighborsClassifier(n_neighbors=5)
    print("[INFO] Loading Dataset")
    kotak, lingkaran, segitiga, garis, segilima, trapesium = load_dataset()
    print("[INFO] Loading Model")
    y_kotak = np.zeros(len(kotak))
    y_lingkaran = np.ones(len(lingkaran))
    y_segitiga = np.ones(len(segitiga)) * 2
    y_garis = np.ones(len(garis)) * 3
    y_segilima = np.ones(len(segilima)) * 4
    y_trapesium = np.ones(len(trapesium)) * 5
    X = kotak + lingkaran + segitiga + garis + segilima + trapesium
    y = np.concatenate([y_kotak, y_lingkaran, y_segitiga, y_garis, y_segilima, y_trapesium])
    model.fit(X, y)
    return model