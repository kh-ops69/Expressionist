import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label
from PIL import Image, ImageTk

class PhotoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Upload or Capture")
        self.root.geometry("500x400")
        
        # Label for instructions
        self.label = tk.Label(root, text="Choose to upload a photo or take a photo from your webcam")
        self.label.pack(pady=10)
        
        # Button to upload photos
        self.upload_button = tk.Button(root, text="Upload Photo", command=self.upload_photo)
        self.upload_button.pack(pady=5)
        
        # Button to capture from webcam
        self.capture_button = tk.Button(root, text="Take Photo from Webcam", command=self.take_photo)
        self.capture_button.pack(pady=5)
        
        # Label to display the selected image
        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        # To hold uploaded image path
        self.image_path = None

    def upload_photo(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp")]
        self.image_path = filedialog.askopenfilename(title="Choose a photo", filetypes=filetypes)
        
        if self.image_path:
            self.display_image(self.image_path)
    
    def take_photo(self):
        # Placeholder for webcam capture function
        # Call your webcam capture code here and save the image path to self.image_path
        # For example: self.image_path = your_webcam_function()
        messagebox.showinfo("Capture Photo", "This will trigger the webcam capture function.")
        # Uncomment to display a sample image after capture for testing:
        # self.display_image("path/to/your/captured/photo.jpg")

    def display_image(self, img_path):
        # Open and resize the image
        image = Image.open(img_path)
        image = image.resize((300, 300))
        img = ImageTk.PhotoImage(image)
        
        # Update label with new image
        self.image_label.config(image=img)
        self.image_label.image = img  # Keep reference to avoid garbage collection
    
    def get_uploaded_image_path(self):
        # Returns the path of the uploaded image

        return self.image_path

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoApp(root)
    root.mainloop()