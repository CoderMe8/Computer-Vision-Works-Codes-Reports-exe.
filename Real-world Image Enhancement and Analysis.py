import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Enhancement Tool (Grayscale Edition)")
        self.root.geometry("1200x700") 

        self.original_image = None
        self.processed_image = None
        self.display_original = None
        self.display_processed = None

        self.create_widgets()

    def create_widgets(self):
        # === Menu ===
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Image", command=self.open_image)
        filemenu.add_command(label="Save Processed Image", command=self.save_image)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        # === Layout ===
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Left Panel (Images + Histograms)
        left_panel = tk.Frame(main_frame)
        left_panel.pack(side="left", fill="both", expand=True)

        img_frame = tk.Frame(left_panel)
        img_frame.pack(fill="both", expand=True)

        self.orig_label = tk.Label(img_frame, text="Original Image")
        self.orig_label.pack(side="left", expand=True, padx=5, pady=5)

        self.proc_label = tk.Label(img_frame, text="Processed Image")
        self.proc_label.pack(side="right", expand=True, padx=5, pady=5)

        # Histogram Section
        self.fig = Figure(figsize=(5, 2), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=left_panel)
        self.canvas_hist.get_tk_widget().pack(fill="x")

        # === Right Panel (Controls) with Scrollbar ===
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side="right", fill="y")

        canvas = tk.Canvas(control_frame, width=350)
        scrollbar = tk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.control_frame = scrollable_frame  # everything else attaches here

        # === Enhancement Controls ===
        tk.Label(self.control_frame, text="Enhancements", font=("Arial", 14, "bold")).pack(pady=5)

        # Grayscale
        tk.Button(self.control_frame, text="Grayscale", command=self.to_grayscale).pack(fill="x", padx=5, pady=2)

        # Negative
        tk.Button(self.control_frame, text="Negative", command=self.negative_image).pack(fill="x", padx=5, pady=2)

        # Log Transform
        log_frame = tk.LabelFrame(self.control_frame, text="Log Transform")
        log_frame.pack(fill="x", padx=5, pady=5)
        self.log_c = tk.DoubleVar(value=1.0)
        self.log_inverse = tk.BooleanVar(value=False)
        tk.Scale(log_frame, from_=0.1, to=5, resolution=0.1,
                 orient="horizontal", label="c", variable=self.log_c).pack(fill="x")
        tk.Checkbutton(log_frame, text="Inverse", variable=self.log_inverse).pack(anchor="w")
        tk.Button(log_frame, text="Apply", command=self.log_transform).pack(fill="x", pady=2)

        # Gamma Correction
        gamma_frame = tk.LabelFrame(self.control_frame, text="Gamma Correction")
        gamma_frame.pack(fill="x", padx=5, pady=5)
        self.gamma_val = tk.DoubleVar(value=1.0)
        tk.Scale(gamma_frame, from_=0.1, to=5, resolution=0.1,
                 orient="horizontal", label="Gamma", variable=self.gamma_val).pack(fill="x")
        tk.Button(gamma_frame, text="Apply", command=self.gamma_correction).pack(fill="x", pady=2)

        # Contrast Stretching
        tk.Button(self.control_frame, text="Contrast Stretching",
                  command=self.contrast_stretching).pack(fill="x", padx=5, pady=2)

        # Histogram Equalization
        tk.Button(self.control_frame, text="Histogram Equalization",
                  command=self.hist_equalization).pack(fill="x", padx=5, pady=2)

        # Combo (Fresh + Stacked)
        combo_frame = tk.LabelFrame(self.control_frame, text="Combo Log + Contrast Stretching")
        combo_frame.pack(fill="x", padx=5, pady=5)
        tk.Button(combo_frame, text="Apply Combo (Fresh)", command=self.combo_log_cs_fresh).pack(fill="x", pady=2)
        tk.Button(combo_frame, text="Apply Combo (Stacked)", command=self.combo_log_cs_stacked).pack(fill="x", pady=2)

        # Reset
        tk.Button(self.control_frame, text="Reset to Original", command=self.reset_image).pack(fill="x", padx=5, pady=10)

    # === File Operations ===
    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.processed_image = self.original_image.copy()
            self.update_display()

    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("BMP files", "*.bmp")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)

    # === Enhancements ===
    def to_grayscale(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_display()

    def negative_image(self):
        if self.processed_image is not None:
            self.processed_image = 255 - self.processed_image
            self.update_display()

    def log_transform(self):
        if self.processed_image is not None:
            c = self.log_c.get()
            img = self.processed_image / 255.0
            if self.log_inverse.get():
                img = np.exp(img * c) - 1
            else:
                img = c * np.log1p(img)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = img.astype(np.uint8)
            self.update_display()

    def gamma_correction(self):
        if self.processed_image is not None:
            gamma = self.gamma_val.get()
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
            self.processed_image = cv2.LUT(self.processed_image, table)
            self.update_display()

    def contrast_stretching(self):
        if self.processed_image is not None:
            min_val = np.min(self.processed_image)
            max_val = np.max(self.processed_image)
            self.processed_image = ((self.processed_image - min_val) /
                                    (max_val - min_val) * 255).astype(np.uint8)
            self.update_display()

    def hist_equalization(self):
        if self.processed_image is not None:
            self.processed_image = cv2.equalizeHist(self.processed_image)
            self.update_display()

    def combo_log_cs_fresh(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.log_transform()
            self.contrast_stretching()

    def combo_log_cs_stacked(self):
        if self.processed_image is not None:
            self.log_transform()
            self.contrast_stretching()

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_display()

    # === Display Update ===
    def update_display(self):
        if self.original_image is not None:
            img = Image.fromarray(self.original_image)
            img = img.resize((350, 350))
            self.display_original = ImageTk.PhotoImage(img)
            self.orig_label.config(image=self.display_original, text="")

        if self.processed_image is not None:
            img = Image.fromarray(self.processed_image)
            img = img.resize((350, 350))
            self.display_processed = ImageTk.PhotoImage(img)
            self.proc_label.config(image=self.display_processed, text="")

            # Update histograms
            self.ax1.clear()
            self.ax2.clear()
            self.ax1.hist(self.original_image.ravel(), 256, [0, 256])
            self.ax1.set_title("Original Histogram")
            self.ax2.hist(self.processed_image.ravel(), 256, [0, 256])
            self.ax2.set_title("Processed Histogram")
            self.canvas_hist.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
