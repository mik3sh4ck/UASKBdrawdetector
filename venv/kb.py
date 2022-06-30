import tkinter as tk
import os
import ai
import numpy as np
from PIL import Image, ImageTk , ImageDraw

model = ai.load_ai()
window = tk.Tk()


img = Image.new(mode="1", size=(500, 500), color=0)
tkimage = ImageTk.PhotoImage(img)
canvas = tk.Label(window, image=tkimage)
canvas.pack()

draw = ImageDraw.Draw(img)

last_point = (0, 0)
prediction = tk.StringVar()
label = tk.Label(window, textvariable=prediction)

def draw_image(event):
    global last_point, tkimage
    current_point = (event.x, event.y)
    draw.line([last_point,current_point],fill=255,width=50)
    last_point = current_point
    tkimage=ImageTk.PhotoImage(img)
    canvas['image']=tkimage
    canvas.pack()
    img_temp = img.resize((28, 28))
    img_temp = np.array(img_temp)
    img_temp = img_temp.flatten()
    output = model.predict([img_temp])
    if(output[0] == 0):
        prediction.set("kotak")
    elif(output[0] == 1):
        prediction.set("lingkaran")
    elif(output[0] == 2):
        prediction.set("segitiga")
    elif(output[0] == 3):
        prediction.set("garis")
    elif(output[0] == 4):
        prediction.set("segilima")
    label.pack()

def start_draw(event):
    global last_point
    last_point = (event.x, event.y)

def reset_canvas(event):
    global tkimage, img, draw
    img = Image.new(mode="1", size=(500, 500), color=0)
    draw = ImageDraw.Draw(img)
    tkimage = ImageTk.PhotoImage(img)
    canvas['image'] = tkimage
    canvas.pack()

kotak = len(os.listdir("venv/kotak"))
lingkaran = len(os.listdir("venv/lingkaran"))
segitiga = len(os.listdir("venv/segitiga"))
garis = len(os.listdir("venv/garis"))
segilima = len(os.listdir("venv/segilima"))

def save_image(event):
    global kotak, lingkaran, segitiga, garis, segilima
    img_temp = img.resize((28, 28))
    if(event.char == "k"):
        img_temp.save(f"venv/kotak/{kotak}.png")
        kotak += 1
    elif(event.char == "l"):
        img_temp.save(f"venv/lingkaran/{lingkaran}.png")
        lingkaran += 1
    elif(event.char == "s"):
        img_temp.save(f"venv/segitiga/{segitiga}.png")
        segitiga += 1
    elif(event.char == "g"):
        img_temp.save(f"venv/garis/{garis}.png")
        garis += 1
    elif(event.char == "q"):
        img_temp.save(f"venv/segilima/{segilima}.png")
        segilima += 1
    

window.bind("<B1-Motion>", draw_image)
window.bind("<ButtonPress-1>", start_draw)
window.bind("<ButtonPress-3>", reset_canvas)
window.bind("<Key>", save_image)



window.mainloop()