import tkinter as tk
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import model


class Title(tk.Label):
    def __init__(self, parent, content, size=25):
        super().__init__(parent, text=content, font=("Courier", size))


class Window(tk.Tk):
    HEIGHT = 500
    WIDTH = 600

    def __init__(self):
        super().__init__()
        self.geometry(f"{self.HEIGHT}x{self.WIDTH}")
        self.title("Flower Classification")
        self.main_frame = tk.Frame(self, bg="lightblue", height=self.HEIGHT*0.9, width=self.HEIGHT*0.9)
        self.main_frame.place(relx=0.05, rely=0.05)
        self.answer_label = Title(self.main_frame, "", size=15)
        self.answer_label.place(relx=0.05, rely=0.2)

        self.image_label = tk.Label(self.main_frame)
        self.image_label.place(relx=0.05, rely=0.3)
        self.place_widgets()

    def place_widgets(self):
        title = Title(self.main_frame, "Classify flowers")
        title.place(relx=0.05, rely=0.05)

        upload_button = tk.Button(self.main_frame, text="Upload image", command=self.classify_image)
        upload_button.place(relx=0.05, rely=0.15)

    def classify_image(self):
        file_path = askopenfile(mode="r",
                                filetypes=[("Png Files", "*png"), ("Jpg Files", "*jpg"), ("Jpeg files", "*jpeg")])
        if file_path is None:
            return

        self.answer_label.config(text=f"Prediction: {model.classify_new_image(file_path.name)}")
        img = Image.open(file_path.name)
        width, height = img.size
        ratio = width / height
        new_height = 200
        new_width = int(ratio * new_height)
        img = img.resize((new_width, new_height))

        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img


if __name__ == '__main__':
    app = Window()
    app.mainloop()
