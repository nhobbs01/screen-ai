import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
from dataset_utils import get_annotated_image_by_number

class ImageViewer:
    def __init__(self, master, split='train'):
        self.master = master
        self.split = split
        self.current_index = 0
        self.image_count = self.get_image_count()
        
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.prev_button = tk.Button(master, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(master, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT)

        self.show_image()

    def get_image_count(self):
        # Assuming you have a function to get the total number of images
        # For example, you can read the CSV file and return the count
        import pandas as pd
        df = pd.read_csv(f'../screen_annotation/{self.split}.csv')
        return len(df)

    def show_image(self):
        if self.current_index < 0 or self.current_index >= self.image_count:
            messagebox.showinfo("Info", "No more images.")
            return
        # Get the PIL image from the dataset utility
        pil_image = get_annotated_image_by_number(self.current_index, self.split)
        pil_image.thumbnail((500, 500))
        # Convert the PIL image to a format tkinter can use
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update the label to show the image
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image  # Keep a reference to avoid garbage

    def show_previous(self):
        self.current_index -= 1
        self.show_image()

    def show_next(self):
        self.current_index += 1
        self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Viewer")
    viewer = ImageViewer(root, split='train')
    root.mainloop()