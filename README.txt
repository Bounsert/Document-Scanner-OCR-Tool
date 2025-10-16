Document Scanner & OCR Tool

A smart, lightweight, and user-friendly document scanning solution built with Python.

Overview

**Document Scanner & OCR Tool** is a desktop application that allows you to scan, crop, enhance, and save documents from images or a webcam.  
It supports **batch scanning**, **PDF export**, and **OCR text extraction**, making it a complete and accessible tool for students, professionals, and developers.

The application focuses on simplicity and clarity — clean design, intuitive controls, and meaningful features that actually help.



Features

 **Smart Document Detection** – Automatically detects and crops document edges.  
 **Batch Processing** – Scan and process multiple images or entire folders.  
 **PDF Export** – Combine scanned pages into a single PDF with adjustable quality.  
 **OCR (Text Recognition)** – Extract text from documents using `pytesseract`.  
 **Webcam Support** – Capture documents directly from your camera.  
 **Auto-Naming & Logging** – Automatically saves and logs processed files.  
 **Resizable Output** – Adjust image size and quality for optimal results.  
 **Cross-Platform GUI** – Built with `Tkinter`, works on Windows, macOS, and Linux.

Installation

1. Clone or download this repository

2. Install dependencies
pip install opencv-python numpy imutils pillow pytesseract
 
To enable OCR, you must also install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your system and ensure it’s added to your system PATH.

3. Run the application
python scan_app.py

Building an Executable (Optional)

To create a standalone `.exe` version (Windows only):
pip install pyinstaller
pyinstaller --onefile --noconsole scan_app.py
Your executable will appear in the `dist` folder.
1. Launch the application.  
2. Choose an image, folder, or use the webcam.  
3. Adjust settings (enhancement, resize, output quality).  
4. Click **Start Processing** — your scanned files will appear in the output folder.  
5. (Optional) Extract text using OCR or merge files into a single PDF.

Technology Stack

 **Language:** Python 3.10+  
 **Libraries:** OpenCV, NumPy, Pillow, imutils, pytesseract, Tkinter  
 **Export Formats:** JPG, PNG, PDF, TXT  
 **Interface:** Graphical (Tkinter GUI)

A Note from the Developer

This project was created with care to make document scanning more accessible — not only for developers but also for everyday users.  
It’s open for improvements, so feel free to suggest ideas, report bugs, or contribute to the project.

If this tool saves you time — that’s exactly what it was meant to do. 

You are free to use, modify, and distribute it with proper attribution.
https://github.com/Bounsert

