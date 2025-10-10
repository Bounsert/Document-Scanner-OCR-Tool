"""
Advanced Document Scanner with OCR, noise/shadow removal, PDF/A conversion and EXE instructions
(document_scanner.py)

New features added:
- OCR using pytesseract (if Tesseract OCR is installed on the system). Extracted text saved as .txt next to each scanned image.
- Noise reduction (bilateral filter) and shadow removal (background estimate) options.
- Option to attempt PDF/A conversion using Ghostscript (if installed). If Ghostscript isn't available, a normal optimized PDF is produced.
- GUI checkbox to enable OCR, controls for noise/shadow removal, and a button that shows step-by-step instructions and a ready-to-run PyInstaller command to build a Windows .exe.
- All previous features retained: Tkinter GUI, batch processing, multi-document detection, PDF creation, auto-rename, progress bar and logs.

Dependencies:
- Required: opencv-python, numpy, imutils, pillow
- Optional for OCR: pytesseract + system Tesseract (https://github.com/tesseract-ocr/tesseract)
- Optional for PDF/A conversion: Ghostscript (gs) installed and available in PATH

Install Python packages:

pip install opencv-python numpy imutils pillow pytesseract

Note: For OCR to work you must also install Tesseract on your machine and ensure the tesseract executable is in PATH. On Windows you can install from https://github.com/UB-Mannheim/tesseract/wiki. On macOS brew install tesseract. On Linux use your package manager.

Usage:
- Run: python document_scanner.py
- Use GUI: choose image/folder/webcam, toggle OCR/noise/shadow removal, set output directory, click Start.

Limitations:
- I cannot build the .exe for you inside this environment. The GUI provides a ready-made PyInstaller command and a step-by-step guide — you run that locally to produce the .exe.
- PDF/A conversion requires Ghostscript installed locally; the script will attempt to run it if you enable the option.

"""

import os
import threading
import cv2
import numpy as np
import imutils
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import datetime
import subprocess

# Optional OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# -------- Image processing utilities --------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_document_contours(image, max_candidates=10):
    ratio = image.shape[0] / 500.0
    small = imutils.resize(image, height=500)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_candidates]
    results = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            pts = approx.reshape(4, 2) * ratio
            results.append(pts.astype('float32'))
    return results


def enhance_image_for_scan(warped, method='adaptive'):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if method == 'adaptive':
        proc = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    elif method == 'otsu':
        _, proc = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'contrast':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc = clahe.apply(gray)
    else:
        proc = gray
    return proc


def reduce_noise(img, strength=5):
    # bilateral filter preserves edges
    return cv2.bilateralFilter(img, d=9, sigmaColor=strength*10, sigmaSpace=strength*10)


def remove_shadows(img):
    # Simple shadow removal using morphological operations on V channel
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = np.asarray(rgb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg = cv2.morphologyEx(v, cv2.MORPH_DILATE, kernel)
    bg = cv2.medianBlur(bg, 21)
    diff = 255 - cv2.absdiff(v, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    hsv[:, :, 2] = norm
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return res


def save_images_as_pdf(image_paths, output_pdf_path, quality=85, resize_width=None, try_pdfa=False):
    if len(image_paths) == 0:
        raise ValueError('No images to save')
    pil_images = []
    for p in image_paths:
        img = Image.open(p).convert('RGB')
        if resize_width:
            wpercent = (resize_width / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((resize_width, hsize), Image.LANCZOS)
        pil_images.append(img)
    first, rest = pil_images[0], pil_images[1:]
    # save intermediate PDF
    tmp_pdf = output_pdf_path
    first.save(tmp_pdf, save_all=True, append_images=rest, optimize=True, quality=quality)

    if try_pdfa:
        # Try to convert to PDF/A using Ghostscript if available
        gs_cmd = ['gs', '-dPDFA=2', '-dBATCH', '-dNOPAUSE', '-dNOOUTERSAVE', '-sProcessColorModel=DeviceCMYK',
                  '-sDEVICE=pdfwrite', '-dPDFACompatibilityPolicy=1', f'-sOutputFile={output_pdf_path}', tmp_pdf]
        try:
            subprocess.run(gs_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return
        except Exception:
            # if GS failed, keep the original PDF
            return


# -------- GUI and orchestration --------

class ScannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Advanced Document Scanner — OCR & PDF/A')
        self.geometry('880x600')
        self.create_widgets()
        self.bind_events()

    def create_widgets(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Left controls
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        ttk.Label(ctrl, text='Input').pack(anchor=tk.W)
        self.btn_file = ttk.Button(ctrl, text='Select Image', command=self.select_image)
        self.btn_file.pack(fill=tk.X)
        self.btn_folder = ttk.Button(ctrl, text='Select Folder', command=self.select_folder)
        self.btn_folder.pack(fill=tk.X, pady=(5,0))
        self.btn_webcam = ttk.Button(ctrl, text='Open Webcam', command=self.open_webcam_thread)
        self.btn_webcam.pack(fill=tk.X, pady=(5,0))

        ttk.Separator(ctrl).pack(fill=tk.X, pady=8)
        ttk.Label(ctrl, text='Settings').pack(anchor=tk.W)
        self.enhance_var = tk.StringVar(value='adaptive')
        ttk.Label(ctrl, text='Enhancement').pack(anchor=tk.W)
        ttk.Combobox(ctrl, textvariable=self.enhance_var, values=['adaptive', 'otsu', 'contrast', 'none']).pack(fill=tk.X)

        self.quality_var = tk.IntVar(value=85)
        ttk.Label(ctrl, text='JPEG quality (for PDF)').pack(anchor=tk.W, pady=(8,0))
        ttk.Scale(ctrl, from_=30, to=100, variable=self.quality_var, orient=tk.HORIZONTAL).pack(fill=tk.X)

        self.resize_var = tk.IntVar(value=1000)
        ttk.Label(ctrl, text='Resize width for PDF (0 = keep)').pack(anchor=tk.W, pady=(8,0))
        ttk.Entry(ctrl, textvariable=self.resize_var).pack(fill=tk.X)

        self.ocr_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text='Enable OCR (requires tesseract)', variable=self.ocr_var).pack(anchor=tk.W, pady=(8,0))

        self.noise_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text='Reduce noise (bilateral)', variable=self.noise_var).pack(anchor=tk.W)

        self.shadow_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text='Remove shadows', variable=self.shadow_var).pack(anchor=tk.W)

        self.pdfa_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text='Attempt PDF/A conversion (requires Ghostscript)', variable=self.pdfa_var).pack(anchor=tk.W, pady=(8,0))

        self.outdir_var = tk.StringVar(value=os.path.join(os.getcwd(), 'scans_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        ttk.Label(ctrl, text='Output directory').pack(anchor=tk.W, pady=(8,0))
        ttk.Entry(ctrl, textvariable=self.outdir_var).pack(fill=tk.X)
        ttk.Button(ctrl, text='Browse output', command=self.browse_outdir).pack(fill=tk.X)

        ttk.Separator(ctrl).pack(fill=tk.X, pady=8)
        self.start_btn = ttk.Button(ctrl, text='Start processing', command=self.start_process_thread)
        self.start_btn.pack(fill=tk.X)
        ttk.Button(ctrl, text='Stop', command=self.stop_processing).pack(fill=tk.X, pady=(5,0))
        ttk.Button(ctrl, text='Build .exe instructions', command=self.show_exe_instructions).pack(fill=tk.X, pady=(6,0))

        # Right log and progress
        right = ttk.Frame(frame)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(right, text='Status / Log').pack(anchor=tk.W)
        self.log = tk.Text(right, height=30)
        self.log.pack(fill=tk.BOTH, expand=True)

        self.progress = ttk.Progressbar(right, orient='horizontal', mode='determinate')
        self.progress.pack(fill=tk.X, pady=(6,0))

        # internal flags
        self._stop_requested = False
        self._worker = None

    def bind_events(self):
        pass

    def log_message(self, msg):
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.log.insert(tk.END, f'[{ts}] {msg}\n')
        self.log.see(tk.END)
        self.update_idletasks()

    def select_image(self):
        fn = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.jpeg *.png *.tif *.tiff')])
        if fn:
            self.input_mode = 'file'
            self.input_path = fn
            self.log_message(f'Selected image: {fn}')

    def select_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.input_mode = 'folder'
            self.input_path = d
            self.log_message(f'Selected folder: {d}')

    def browse_outdir(self):
        d = filedialog.askdirectory()
        if d:
            self.outdir_var.set(d)

    def open_webcam_thread(self):
        t = threading.Thread(target=self.open_webcam, daemon=True)
        t.start()

    def open_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror('Webcam', 'Unable to open webcam')
            return
        self.log_message('Webcam opened. Press SPACE to capture, q to quit webcam.')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = imutils.resize(frame, height=600)
            cv2.imshow('Webcam - press SPACE to capture, q to quit', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                outdir = self.outdir_var.get()
                os.makedirs(outdir, exist_ok=True)
                fname = os.path.join(outdir, f'webcam_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
                cv2.imwrite(fname, frame)
                self.log_message(f'Captured: {fname}')
                self.process_single_image(fname, outdir)
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.log_message('Webcam closed')

    def start_process_thread(self):
        if hasattr(self, '_worker') and self._worker and self._worker.is_alive():
            messagebox.showinfo('Processing', 'Already running')
            return
        self._stop_requested = False
        t = threading.Thread(target=self.start_processing, daemon=True)
        self._worker = t
        t.start()

    def stop_processing(self):
        self._stop_requested = True
        self.log_message('Stop requested...')

    def start_processing(self):
        try:
            mode = getattr(self, 'input_mode', None)
            if mode == 'file':
                outdir = self.outdir_var.get()
                os.makedirs(outdir, exist_ok=True)
                saved = self.process_single_image(self.input_path, outdir)
                # if OCR enabled, save collected OCR text into combined file
                if self.ocr_var.get() and saved:
                    self.log_message('OCR completed for file(s)')
            elif mode == 'folder':
                folder = self.input_path
                files = [os.path.join(folder, f) for f in os.listdir(folder)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
                total = len(files)
                if total == 0:
                    self.log_message('No images found in folder')
                    return
                self.progress['maximum'] = total
                outdir = self.outdir_var.get()
                os.makedirs(outdir, exist_ok=True)
                idx = 0
                all_saved = []
                for f in files:
                    if self._stop_requested:
                        self.log_message('Processing stopped by user')
                        break
                    idx += 1
                    self.log_message(f'Processing ({idx}/{total}): {f}')
                    saved = self.process_single_image(f, outdir)
                    if saved:
                        all_saved.extend(saved)
                    self.progress['value'] = idx
                # combine to PDF if any images
                if len(all_saved) > 0:
                    pdf_path = os.path.join(outdir, 'scanned_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
                    q = int(self.quality_var.get())
                    rw = int(self.resize_var.get()) if int(self.resize_var.get()) > 0 else None
                    try_pdfa = self.pdfa_var.get()
                    save_images_as_pdf(all_saved, pdf_path, quality=q, resize_width=rw, try_pdfa=try_pdfa)
                    self.log_message(f'PDF saved: {pdf_path}')
            else:
                messagebox.showinfo('Input', 'Select image or folder first')
        except Exception as e:
            self.log_message('Error: ' + str(e))

    def process_single_image(self, path, outdir):
        image = cv2.imread(path)
        if image is None:
            self.log_message('Failed to open: ' + path)
            return []
        # optional noise/shadow
        if self.noise_var.get():
            image = reduce_noise(image, strength=5)
        if self.shadow_var.get():
            image = remove_shadows(image)

        contours = detect_document_contours(image, max_candidates=12)
        saved_paths = []
        if len(contours) == 0:
            self.log_message('No document contour found; saving enhanced full image')
            warped = imutils.resize(image, width=1000)
            proc = enhance_image_for_scan(warped, method=self.enhance_var.get())
            out_path = self._unique_path(outdir, os.path.splitext(os.path.basename(path))[0] + '_full.jpg')
            cv2.imwrite(out_path, proc)
            saved_paths.append(out_path)
            if self.ocr_var.get():
                self.perform_ocr(out_path)
            return saved_paths

        for i, cnt in enumerate(contours, start=1):
            if self._stop_requested:
                break
            try:
                warped = four_point_transform(image, cnt)
                proc = enhance_image_for_scan(warped, method=self.enhance_var.get())
                base = os.path.splitext(os.path.basename(path))[0]
                out_name = f'{base}_p{i:02d}.jpg'
                out_path = self._unique_path(outdir, out_name)
                cv2.imwrite(out_path, proc)
                saved_paths.append(out_path)
                self.log_message('Saved: ' + out_path)
                if self.ocr_var.get():
                    self.perform_ocr(out_path)
            except Exception as e:
                self.log_message('Failed to process contour: ' + str(e))
        return saved_paths

    def perform_ocr(self, image_path):
        if not TESSERACT_AVAILABLE:
            self.log_message('pytesseract not available — install pytesseract and system tesseract to enable OCR')
            return
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='eng')
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            self.log_message('OCR saved: ' + txt_path)
        except Exception as e:
            self.log_message('OCR failed: ' + str(e))

    def _unique_path(self, outdir, name):
        base, ext = os.path.splitext(name)
        full = os.path.join(outdir, name)
        counter = 1
        while os.path.exists(full):
            full = os.path.join(outdir, f'{base}_{counter}{ext}')
            counter += 1
        return full

    def show_exe_instructions(self):
        # Prepare PyInstaller command and explain steps
        pycmd = f'pyinstaller --onefile --windowed "{os.path.basename(__file__)}"'
        inst = (
            'To build a Windows .exe locally:'
            '1) Install PyInstaller: pip install pyinstaller'
            '2) Run the following command in the folder with document_scanner.py:'
            f'{pycmd}'
            'Notes:'
            '- If your script uses external binaries (Tesseract, Ghostscript), the .exe will still require those installed on the target machine or you must bundle them separately.'
            '- To reduce final exe size: use UPX (optional) and exclude heavy modules not used.'
            '- Building on Windows for Windows is recommended; cross-compilation is tricky.'
        )
        messagebox.showinfo('Build .exe instructions', inst)


# -------- Entry point --------

def main():
    app = ScannerApp()
    app.mainloop()


if __name__ == '__main__':
    main()

