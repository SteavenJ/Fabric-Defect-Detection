import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
from skimage.feature import graycomatrix, graycoprops

# ==============================================================================
# KONFIGURASI PARAMETER
# ==============================================================================
LIMIT_CONTRAST = 1100.0  
LIMIT_INTENSITY_DIFF = 30.0 
LIMIT_ANGLE_DEVIATION = 10.0 

class FabricFixedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi Kain (Fixed Layout)")
        self.root.geometry("1100x700") # Ukuran Jendela Tetap
        self.root.resizable(False, False) # KUNCI Jendela agar tidak bisa ditarik
        self.root.configure(bg="#222831")

        # --- HEADER ---
        tk.Label(root, text="SISTEM DETEKSI TEKSTUR & POLA (SMART FIT)", 
                 font=("Segoe UI", 18, "bold"), bg="#222831", fg="#00adb5").pack(pady=15)
        
        # --- TOMBOL INPUT ---
        self.btn_load = tk.Button(root, text="📂 INPUT GAMBAR", command=self.upload_image, 
                                  font=("Segoe UI", 11, "bold"), bg="#393e46", fg="white", 
                                  width=25, height=2, relief="flat", cursor="hand2")
        self.btn_load.pack(pady=5)

        # --- PANEL UTAMA (GRID 3 KOLOM) ---
        # Kita kunci ukuran frame utama agar rapi
        self.frame_main = tk.Frame(root, bg="#222831")
        self.frame_main.pack(expand=True, fill="both", padx=10, pady=10)

        # Helper untuk membuat panel yang ukurannya TERKUNCI
        def create_fixed_panel(parent, title):
            # Container Frame
            outer = tk.Frame(parent, bg="#393e46", bd=2, relief="groove", width=340, height=300)
            outer.pack_propagate(False) # PENTING: Mencegah frame membesar mengikuti isi
            
            # Judul
            tk.Label(outer, text=title, bg="#393e46", fg="#eeeeee", 
                     font=("Arial", 10, "bold")).pack(fill="x", pady=5)
            
            # Label Gambar (Ditengah)
            lbl = tk.Label(outer, bg="black")
            lbl.pack(expand=True, fill="both", padx=5, pady=5)
            
            return outer, lbl

        # 1. Citra Asli
        self.fr_1, self.lbl_orig = create_fixed_panel(self.frame_main, "1. Citra Asli")
        self.fr_1.pack(side="left", padx=10)
        
        # 2. Lokalisasi
        self.fr_2, self.lbl_loc = create_fixed_panel(self.frame_main, "2. Deteksi Cacat")
        self.fr_2.pack(side="left", padx=10)

        # 3. Pola
        self.fr_3, self.lbl_pat = create_fixed_panel(self.frame_main, "3. Analisis Pola")
        self.fr_3.pack(side="left", padx=10)

        # --- FOOTER STATUS ---
        self.lbl_result = tk.Label(root, text="STATUS: MENUNGGU INPUT", 
                                   font=("Segoe UI", 14, "bold"), bg="#222831", fg="#777")
        self.lbl_result.pack(pady=5)
        
        self.lbl_detail = tk.Label(root, text="-", font=("Consolas", 10), 
                                   bg="#222831", fg="#aaa")
        self.lbl_detail.pack(pady=(0, 20))

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.jpg *.png *.bmp")])
        if not path: return
        
        self.lbl_result.config(text="MEMPROSES...", fg="#f1c40f")
        self.root.update()
        self.process_pipeline(path)

    # ==========================================================================
    # FUNGSI SMART RESIZE (KUNCI UTAMA SOLUSI INI)
    # ==========================================================================
    def display_smart_fit(self, cv_img, label):
        h, w = cv_img.shape[:2]
        
        # Batas Maksimal Tampilan (Pixel) - Sesuaikan dengan ukuran Frame
        MAX_W = 320
        MAX_H = 260
        
        # Hitung Skala berdasarkan Lebar DAN Tinggi
        scale_w = MAX_W / w
        scale_h = MAX_H / h
        
        # Ambil skala terkecil agar gambar muat sepenuhnya (Aspect Fit)
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img_res = cv2.resize(cv_img, (new_w, new_h))
        img_pil = Image.fromarray(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label.config(image=img_tk)
        label.image = img_tk # Simpan referensi

    # ==========================================================================
    # LOGIKA UTAMA (SAMA)
    # ==========================================================================
    def process_pipeline(self, path):
        img_bgr = cv2.imread(path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        if img_gray.dtype != 'uint8':
            img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            
        img_loc_vis = img_bgr.copy()
        img_pat_vis = img_bgr.copy()
        
        h, w = img_gray.shape
        
        # A. WADDING & NEPS
        bg_blur = cv2.GaussianBlur(img_gray, (51, 51), 0)
        diff_map = cv2.absdiff(img_gray, bg_blur)
        
        PATCH = 64
        cnt_wadding = 0
        cnt_neps = 0
        
        for y in range(0, h, PATCH):
            for x in range(0, w, PATCH):
                patch_gray = img_gray[y:min(y+PATCH, h), x:min(x+PATCH, w)]
                patch_diff = diff_map[y:min(y+PATCH, h), x:min(x+PATCH, w)]
                
                if patch_gray.shape[0] < 30 or patch_gray.shape[1] < 30: continue
                
                if np.mean(patch_diff) > LIMIT_INTENSITY_DIFF:
                    cnt_wadding += 1
                    cv2.rectangle(img_loc_vis, (x, y), (x+PATCH, y+PATCH), (0, 255, 255), 2)
                else:
                    glcm = graycomatrix(patch_gray, [1], [0], 256, symmetric=True, normed=True)
                    contrast = graycoprops(glcm, 'contrast')[0, 0]
                    if contrast > LIMIT_CONTRAST:
                        cnt_neps += 1
                        cv2.rectangle(img_loc_vis, (x, y), (x+PATCH, y+PATCH), (0, 0, 255), 2)

        # B. HOUGH TRANSFORM
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=100, maxLineGap=10)
        
        angle_dev = 0.0
        pattern_status = "LURUS"
        
        if lines is not None and len(lines) > 5:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Menebalkan garis agar terlihat di preview kecil
                cv2.line(img_pat_vis, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                deg = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
                if deg > 90: deg -= 180
                angles.append(deg)
            
            angle_dev = np.std(angles)
            if angle_dev > LIMIT_ANGLE_DEVIATION:
                pattern_status = "MIRING"
                # Font diperbesar agar terbaca di preview kecil
                cv2.putText(img_pat_vis, "POLA RUSAK", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)

        # C. KEPUTUSAN
        is_bad = (cnt_neps > 0) or (cnt_wadding > 0) or (pattern_status != "LURUS")
        
        status_txt = "DITERIMA" if not is_bad else "DITOLAK"
        status_col = "#00adb5" if not is_bad else "#ff2e63"
        
        reasons = []
        if cnt_wadding > 0: reasons.append(f"Wadding ({cnt_wadding})")
        if cnt_neps > 0: reasons.append(f"Neps ({cnt_neps})")
        if pattern_status != "LURUS": reasons.append(f"Pola Miring ({angle_dev:.1f})")
        
        # DISPLAY SMART FIT
        self.display_smart_fit(img_bgr, self.lbl_orig)
        self.display_smart_fit(img_loc_vis, self.lbl_loc)
        self.display_smart_fit(img_pat_vis, self.lbl_pat)
        
        self.lbl_result.config(text=f"KEPUTUSAN: {status_txt}", fg=status_col)
        self.lbl_detail.config(text=" | ".join(reasons) if reasons else "Kualitas Kain Sempurna")

if __name__ == "__main__":
    root = tk.Tk()
    app = FabricFixedApp(root)
    root.mainloop()