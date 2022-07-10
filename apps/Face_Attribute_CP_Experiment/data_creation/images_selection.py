import os

from PIL import ImageTk
import tkinter as tk
import pickle as pkl


##Tkinter app

source_filename = './out/random_faces.pkl'
output_filename =  './out/selected_images.pkl'

if os.path.exists(output_filename):
    os.remove(output_filename)

def pickleLoader(pklFile):
    try:
        while True:
            yield pkl.load(pklFile)
    except EOFError:
        pass


def next_image():
    window.seen_counter += 1
    label.config(text="Number of images SEEN : " + str(window.seen_counter))
    var.set(1)


def add_image():
    window.added_counter += 1
    window.seen_counter += 1
    with open(output_filename, 'ab') as output_file:
        pkl.dump(image, output_file)
    label.config(text="Number of images SEEN : " + str(window.seen_counter))
    label1.config(text="Number of images SELECTED : " + str(window.added_counter))
    var.set(1)


def start():
    global label,label1
    label = tk.Label(text="Number of images SEEN : " + str(window.seen_counter))
    label.pack()

    label1 = tk.Label(text="Number of images SELECTED : " + str(window.added_counter))
    label1.pack()

    label2 = tk.Label(window, text="")
    label2.pack()

    button = tk.Button(window, text="Next", command=next_image)
    button.pack()
    window.bind("<q>", lambda event:next_image())

    button1 = tk.Button(window, text="Select Image", command=add_image)
    button1.pack()
    window.bind("<p>", lambda event:add_image())


    startBut.destroy()

    with open(source_filename, 'rb') as file:
        global image
        for image in pickleLoader(file):
            img = ImageTk.PhotoImage(image['image'])
            label2.config(image=img)

            print("Seed: " + str(image['seed']) + "   Index: " + str(image['index_in_seed']))
            button.wait_variable(var)

    print("END")

    button.destroy()
    button1.destroy()
    window.unbind("<q>")
    window.unbind("<p>")

    label3 = tk.Label(text="END")
    label3.pack()



window = tk.Tk()
window.title("Image Selection")

window.added_counter = 0
window.seen_counter = 0

var = tk.IntVar()
var.set(0)

startBut = tk.Button(window, text= "Start", command=start)
startBut.pack()

tk.mainloop()

