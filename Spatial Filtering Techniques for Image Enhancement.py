import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class ImageProcessorApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Interactive Image Processing — Spatial Filtering")
        self.geometry("1200x820")
        self.minsize(900, 600)

        self.original_image = None
        self.processed_image = None
        self.last_transform_text = tk.StringVar(value="Load an image to begin…")
        self._resize_job = None
        self.stack_var = tk.BooleanVar(value=False)  # NEW: stacking option

        self.setup_ui()

    # ----------------------------- UI LAYOUT
    def setup_ui(self):
        # === IMAGE PANES ===
        top = tk.Frame(self, bg="#2c3e50")
        top.pack(side="top", padx=10, pady=(10, 6), fill="both", expand=True)

        pane = tk.PanedWindow(top, orient=tk.HORIZONTAL, sashrelief='raised', bg="#2c3e50")
        pane.pack(fill="both", expand=True)

        left_panel = tk.Frame(pane, bg="#2c3e50", width=600, height=360)
        right_panel = tk.Frame(pane, bg="#2c3e50", width=600, height=360)
        pane.add(left_panel)
        pane.add(right_panel)

        self.orig_label = tk.Label(left_panel, bg="#2c3e50")
        self.orig_label.pack(fill="both", expand=True)
        self.proc_label = tk.Label(right_panel, bg="#2c3e50")
        self.proc_label.pack(fill="both", expand=True)

        # === HISTOGRAMS ===
        mid = tk.Frame(self, bg="#243447")
        mid.pack(side="top", padx=10, pady=(6, 6), fill="x")

        self.fig = Figure(figsize=(10, 2.8), dpi=100)
        self.ax_orig = self.fig.add_subplot(1, 2, 1)
        self.ax_proc = self.fig.add_subplot(1, 2, 2)
        self.ax_orig.set_title("Original Histogram")
        self.ax_proc.set_title("Enhanced Histogram")

        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        self.canvas.get_tk_widget().pack(fill="x", expand=False)

        # === BOTTOM CONTROLS ===
        bottom = tk.Frame(self, bg="#34495e", padx=10, pady=10)
        bottom.pack(side="bottom", fill="x")

        file_frame = tk.Frame(bottom, bg="#34495e")
        file_frame.grid(row=0, column=0, sticky="w", padx=(0, 10))

        tk.Button(file_frame, text="Open Image…", command=self.open_image,
                  bg="#1abc9c", fg="white", font=("Helvetica", 12, "bold")).pack(side="left", padx=5)
        tk.Button(file_frame, text="Reset", command=self.reset_image,
                  bg="#e74c3c", fg="white", font=("Helvetica", 12, "bold")).pack(side="left", padx=5)
        tk.Button(file_frame, text="Save As…", command=self.save_image,
                  bg="#9b59b6", fg="white", font=("Helvetica", 12, "bold")).pack(side="left", padx=5)

        # NEW: Checkbox for stacking
        tk.Checkbutton(file_frame, text="Stack Filters", variable=self.stack_var,
                       bg="#34495e", fg="white").pack(side="left", padx=10)

        # === TRANSFORMATIONS ===
        trans_container = tk.Frame(bottom, bg="#34495e")
        trans_container.grid(row=0, column=1, sticky="ew")
        bottom.grid_columnconfigure(1, weight=1)

        trans_canvas = tk.Canvas(trans_container, bg="#34495e", highlightthickness=0, height=140)
        trans_scroll = ttk.Scrollbar(trans_container, orient="horizontal", command=trans_canvas.xview)
        self.trans_inner = tk.Frame(trans_canvas, bg="#34495e")
        self.trans_inner.bind(
            "<Configure>", lambda e: trans_canvas.configure(scrollregion=trans_canvas.bbox("all"))
        )
        trans_canvas.create_window((0, 0), window=self.trans_inner, anchor="nw")
        trans_canvas.configure(xscrollcommand=trans_scroll.set)
        trans_canvas.pack(fill="x", expand=True)
        trans_scroll.pack(fill="x", expand=False)

        # === Sharpening Filters ===
        sharp_frame = tk.LabelFrame(self.trans_inner, text="Sharpening Filters", bg="#34495e", fg="white")
        sharp_frame.pack(side="left", padx=6)

        tk.Button(sharp_frame, text="Sobel (1st Derivative)",
                  command=lambda: self.apply_transformation("sobel"),
                  bg="#2980b9", fg="white").pack(pady=2, padx=5, fill="x")

        tk.Button(sharp_frame, text="Laplacian (2nd Derivative)",
                  command=lambda: self.apply_transformation("laplacian"),
                  bg="#2980b9", fg="white").pack(pady=2, padx=5, fill="x")

        tk.Button(sharp_frame, text="Sobel + Laplacian",
                  command=lambda: self.apply_transformation("sobel_laplacian"),
                  bg="#16a085", fg="white").pack(pady=2, padx=5, fill="x")

        # === Smoothing Filters ===
        smooth_frame = tk.LabelFrame(self.trans_inner, text="Smoothing Filters", bg="#34495e", fg="white")
        smooth_frame.pack(side="left", padx=6)

        tk.Button(smooth_frame, text="Mean (Averaging)",
                  command=lambda: self.apply_transformation("mean"),
                  bg="#27ae60", fg="white").pack(pady=2, padx=5, fill="x")

        tk.Button(smooth_frame, text="Median",
                  command=lambda: self.apply_transformation("median"),
                  bg="#27ae60", fg="white").pack(pady=2, padx=5, fill="x")

        tk.Button(smooth_frame, text="Mode",
                  command=lambda: self.apply_transformation("mode"),
                  bg="#27ae60", fg="white").pack(pady=2, padx=5, fill="x")

        # === Transformation report ===
        report_frame = tk.Frame(self, bg="#22313f")
        report_frame.pack(fill="x", padx=10, pady=(6, 10))
        tk.Label(report_frame, text="Transformation Used:", bg="#22313f", fg="white",
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.transform_label = tk.Label(report_frame, textvariable=self.last_transform_text,
                                        bg="#22313f", fg="#ecf0f1", justify="left",
                                        font=("Consolas", 10))
        self.transform_label.pack(fill="x")

        self.bind('<Configure>', self._on_resize)

    # ----------------------------- IO FUNCTIONS
    def open_image(self):
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*"))
        )
        if not filepath:
            return
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        self.original_image = img.copy()
        self.processed_image = img.copy()
        self.update_preview_and_histograms()
        self.last_transform_text.set("Loaded image.")

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("No Image", "There is no enhanced image to save.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Save Enhanced Image As",
            defaultextension=".png",
            filetypes=(("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp"), ("TIFF", "*.tiff"))
        )
        if not save_path:
            return
        cv2.imwrite(save_path, self.processed_image)
        messagebox.showinfo("Saved", f"Enhanced image saved to:\n{save_path}")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_preview_and_histograms()
            self.last_transform_text.set("Reset to original image.")

    # ----------------------------- FILTERS
    def apply_transformation(self, t):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        # NEW: stack mode
        if self.stack_var.get():
            src = self.processed_image.copy()
        else:
            src = self.original_image.copy()

        if t == "sobel":
            sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
            dst = cv2.magnitude(sobelx, sobely)
            dst = np.uint8(np.clip(dst, 0, 255))
            self.last_transform_text.set("Sobel Operator (First Derivative)\nKernel: [-1 0 1]")

        elif t == "laplacian":
            dst = cv2.Laplacian(src, cv2.CV_64F, ksize=3)
            dst = np.uint8(np.clip(np.absolute(dst), 0, 255))
            self.last_transform_text.set("Laplacian Operator (Second Derivative)\nKernel: [0 -1 0; -1 4 -1; 0 -1 0]")

        elif t == "sobel_laplacian":
            sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            lap = cv2.Laplacian(src, cv2.CV_64F, ksize=3)
            combined = cv2.addWeighted(sobel, 0.5, np.absolute(lap), 0.5, 0)
            dst = np.uint8(np.clip(combined, 0, 255))
            self.last_transform_text.set("Combined Sobel + Laplacian for stronger edge detection.")

        elif t == "mean":
            dst = cv2.blur(src, (5, 5))
            self.last_transform_text.set("Mean (Averaging) Filter\nKernel: 1/25 * [5x5 matrix of ones]")

        elif t == "median":
            dst = cv2.medianBlur(src, 5)
            self.last_transform_text.set("Median Filter (non-linear, replaces each pixel with median of neighborhood)")

        elif t == "mode":
            def mode_filter(image, ksize=3):
                pad = ksize // 2
                padded = np.pad(image, pad, mode='edge')
                out = np.zeros_like(image)
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        window = padded[i:i+ksize, j:j+ksize].flatten()
                        mode_val = np.bincount(window).argmax()  # faster than scipy
                        out[i, j] = mode_val
                return out

            dst = mode_filter(src, 3)
            self.last_transform_text.set("Mode Filter (replaces each pixel with most frequent value in neighborhood)")

        else:
            return

        self.processed_image = dst
        self.update_preview_and_histograms()

    # ----------------------------- UTILS
    def _show_image_in_label(self, img, label):
        if img is None:
            label.configure(image='')
            return
        frame_w = label.winfo_width() or 400
        frame_h = label.winfo_height() or 300
        h, w = img.shape[:2]
        scale = min(frame_w / w, frame_h / h)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(resized))
        label.configure(image=img_tk)
        label.image = img_tk

    def update_histograms(self):
        self.ax_orig.clear(); self.ax_proc.clear()
        if self.original_image is not None:
            self.ax_orig.hist(self.original_image.ravel(), bins=256, range=(0, 255))
        if self.processed_image is not None:
            self.ax_proc.hist(self.processed_image.ravel(), bins=256, range=(0, 255))
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def update_preview_and_histograms(self):
        self._show_image_in_label(self.original_image, self.orig_label)
        self._show_image_in_label(self.processed_image, self.proc_label)
        self.update_histograms()

    def _on_resize(self, event):
        if self._resize_job:
            try: self.after_cancel(self._resize_job)
            except Exception: pass
        self._resize_job = self.after(180, self.update_preview_and_histograms)

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()

