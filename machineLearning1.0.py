import tkinter as tk
from PIL import Image, ImageGrab, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tkinter import messagebox
from tkinter import *

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")
        self.canvas1 = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas1.pack(side=tk.LEFT)
        self.canvas2 = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas2.pack(side=tk.LEFT)
        self.drawing1 = False
        self.drawing2 = False
        self.previous_mouse_x1, self.previous_mouse_y1 = 0, 0
        self.previous_mouse_x2, self.previous_mouse_y2 = 0, 0
        self.model = load_model(R"C:\Users\user\OneDrive\programmi\macine learnign\versioni programmi\filecheckpoint1.keras")
        self.bind_events()
        self.create_buttons()

    def bind_events(self):
        self.canvas1.bind('<B1-Motion>', self.draw1)
        self.canvas1.bind('<Button-1>', self.start_drawing1)
        self.canvas1.bind('<ButtonRelease-1>', self.stop_drawing1)
        self.canvas2.bind('<B1-Motion>', self.draw2)
        self.canvas2.bind('<Button-1>', self.start_drawing2)
        self.canvas2.bind('<ButtonRelease-1>', self.stop_drawing2)

    def start_drawing1(self, event):
        self.drawing1 = True
        self.previous_mouse_x1, self.previous_mouse_y1 = event.x, event.y

    def draw1(self, event):
        if self.drawing1:
            self.canvas1.create_line(self.previous_mouse_x1, self.previous_mouse_y1, event.x, event.y, fill='white', width=5)
            self.previous_mouse_x1, self.previous_mouse_y1 = event.x, event.y

    def stop_drawing1(self, event):
        self.drawing1 = False

    def start_drawing2(self, event):
        self.drawing2 = True
        self.previous_mouse_x2, self.previous_mouse_y2 = event.x, event.y

    def draw2(self, event):
        if self.drawing2:
            self.canvas2.create_line(self.previous_mouse_x2, self.previous_mouse_y2, event.x, event.y, fill='white', width=5)
            self.previous_mouse_x2, self.previous_mouse_y2 = event.x, event.y

    def stop_drawing2(self, event):
        self.drawing2 = False

    def clear_canvases(self):
        self.canvas1.delete("all")
        self.canvas2.delete("all")

    def get_image(self, canvas):
        x = self.root.winfo_rootx() + canvas.winfo_x()
        y = self.root.winfo_rooty() + canvas.winfo_y()
        img = ImageGrab.grab(bbox=(x, y, x + 280, y + 280))
        img = img.resize((28, 28))
        img_gray = img.convert('L')
        arr = np.array(img_gray)
        arr[0,:] = 0
        arr[:,0] = 0
        return arr

    def display_drawings(self):
        arr1 = self.get_image(self.canvas1)
        arr2 = self.get_image(self.canvas2)
        plt.subplot(1, 2, 1)
        plt.imshow(arr1, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(arr2, cmap='gray')
        plt.show()

    def create_buttons(self):
        btn_confirm = tk.Button(self.root, text="Confirm", command=self.recognize_drawing)
        btn_confirm.pack()

        btn_clear = tk.Button(self.root, text="Clear Canvases", command=self.clear_canvases)
        btn_clear.pack()

        btn_operation = tk.Button(self.root, text="Operations", command=self.operation)
        btn_operation.pack()


    def preprocess_image(self, arr):
        arr_reshaped = arr.reshape(1, 28, 28)
        arr_scaled = arr_reshaped / 255.0
        arr_tensor = arr_scaled.reshape(1, 28, 28, 1)
        return arr_tensor

    def recognize_drawing(self):
        arr = self.get_image(self.canvas1)
        arr_tensor = self.preprocess_image(arr)
        global predicted_number
        predicted_number = np.argmax(self.model.predict(arr_tensor), axis=1)

        arr1 = self.get_image(self.canvas2)
        arr_tensor1 = self.preprocess_image(arr1)
        global predicted_number1
        predicted_number1 = np.argmax(self.model.predict(arr_tensor1), axis=1)

        self.display_result(arr, arr1)

    def display_result(self, arr, arr1):
        plt.subplot(1, 2, 1)
        plt.imshow(arr, cmap='gray')
        plt.title("Predicted Number: " + str(predicted_number))
        plt.subplot(1, 2, 2)
        plt.imshow(arr1,cmap='gray')
        plt.title("Predicted Number: " + str(predicted_number1))
        plt.show()
    
    def operation(self):
        try:
            predicted_number
            predicted_number1
        except:
            messagebox.showwarning(title=None, message="Effettuare Prima Il Riconoscimento Del Numero Con Il Pulsante Confirm")
        else:
            

            finestra2 = Tk()

            screen_width = finestra2.winfo_screenwidth()
            screen_height = finestra2.winfo_screenheight()

            print(screen_width , "\n", screen_height)
            finestra2.title("risultati operazioni")
            finestra2.geometry("1480x900")
            finestra2.resizable(False,False)
            finestra2.configure(bg = "#FFFFFF")

            canvas = Canvas(
                            finestra2,
                            bg = "grey",
                            height = screen_height,
                            width = screen_width,
                            bd = 0,
                            highlightthickness = 0,
                            relief = "ridge"
                        )

            canvas.place(x = 0, y = 0)

            canvas.create_rectangle(
                            715,#asse x
                            194,#asse y
                            715,#asse x
                            194 + 142,#asse y
                            fill="#000000",
                            outline="")

            canvas.create_rectangle(
                            433,
                            136,
                            433 + 565,
                            136,
                            fill="#000000",
                            outline="")

            canvas.create_text(
                            #I tuoi Numeri
                            578.0,
                            54.0,
                            anchor="nw",
                            text=("Your Numbers"),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )
            canvas.create_text(
                            #Numero 1
                            491,
                            220,
                            anchor="nw",
                            text=(predicted_number),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )
            canvas.create_text(
                            #Numero 2
                            891,
                            220,
                            anchor="nw",
                            text=(predicted_number1),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )

            canvas.create_text(
                            #Somma
                            146,
                            450,
                            anchor="nw",
                            text=("sum"),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )

            canvas.create_text(
                            #Numero Somma
                            177,
                            659,
                            anchor="nw",
                            text=(predicted_number + predicted_number1),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )

            canvas.create_text(
                            #Sottrazione
                            395,
                            450,
                            anchor="nw",
                            text=("Subtraction"),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )

            canvas.create_text(
                            #Numero Sottrazione
                            491,
                            659,
                            anchor="nw",
                            text=(predicted_number - predicted_number1),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )

            canvas.create_text(
                            #Moltiplicazione
                            801,
                            450,
                            anchor="nw",
                            text=("Moltiplication"),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )

            canvas.create_text(
                            #Numero Moltiplicazione
                            905,
                            659,
                            anchor="nw",
                            text=(predicted_number * predicted_number1),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )

            canvas.create_text(
                            #Divisione
                            1207,
                            450,
                            anchor="nw",
                            text=("Division"),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )
            divisione = predicted_number / predicted_number1
            divisione1 = np.around(divisione, 2)
            canvas.create_text(
                            #Numero Divisione
                            1270,
                            659,
                            anchor="nw",
                            text=(divisione1),
                            fill="#000000",
                            font=("Inter Bold", 48 * -1)
                        )

            canvas.create_rectangle(
                            96,
                            545,
                            96 + 1287,
                            546,
                            fill="#000000",
                            outline="")

            canvas.create_rectangle(
                            308,
                            450,
                            309,
                            450 + 348,
                            fill="#000000",
                            outline="")

            canvas.create_rectangle(
                            714,
                            440,
                            714,
                            440 + 348,
                            fill="#000000",
                            outline="")

            canvas.create_rectangle(
                            1136,
                            440,
                            1136,
                            440 + 348,
                            fill="#000000",
                            outline="")

            canvas.create_rectangle(
                            280,
                            517,
                            280 + 57,
                            517 + 57,
                            fill="gray",
                            outline="")

            canvas.create_rectangle(
                            683,
                            517,
                            683 + 57,
                            517 + 57,
                            fill="gray",
                            outline="")

            canvas.create_rectangle(
                            1109,
                            517,
                            1109 + 57,
                            517 + 57,
                            fill="gray",
                            outline="")

            finestra2.mainloop()
            
            print(predicted_number + predicted_number1)
            print(predicted_number - predicted_number1)
            print(predicted_number * predicted_number1)
            print(predicted_number / predicted_number1)

root = tk.Tk()
app = DrawingApp(root)
root.mainloop()

