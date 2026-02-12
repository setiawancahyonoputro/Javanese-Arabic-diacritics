# SCRIPT BERHASIL MEMOTONG HURUF SESUAI DENGAN STUKTUR HURUFNYA PER SUB-PATH





import os
#os.chdir("/shm")
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
from skimage.morphology import skeletonize
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree
import SimpleITK as sitk
from skimage.filters import threshold_local
from skimage.morphology import medial_axis, thin
from scipy.ndimage import distance_transform_edt, maximum_filter


PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)
def freeman(x, y):
    if (y==0):
        y=1e-9 # so that we escape the divby0 exception
    if (x==0):
        x=-1e-9 # biased to the left as the text progresses leftward
    if (abs(x/y)<pow(PHI,2)) and (abs(y/x)<pow(PHI,2)): # corner angles
        if   (x>0) and (y>0):
            return(1)
        elif (x<0) and (y>0):
            return(3)
        elif (x<0) and (y<0):
            return(5)
        elif (x>0) and (y<0):
            return(7)
    else: # square angles
        if   (x>0) and (abs(x)>abs(y)):
            return(int(0))
        elif (y>0) and (abs(y)>abs(x)):
            return(2)
        elif (x<0) and (abs(x)>abs(y)):
            return(4)
        elif (y<0) and (abs(y)>abs(x)):
            return(6)
        
RESIZE_FACTOR=2
SLIC_SPACE= 3
SLIC_SPACE= SLIC_SPACE*RESIZE_FACTOR

THREVAL= 60
RASMVAL= 160

CHANNEL= 2





def draw(img): # draw the bitmap
    plt.figure(dpi=600)
    plt.grid(False)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        
        
def dt_ridge_skeleton(fg_bool, r_min=1.0, neigh=3):
    """
    Pure DT-based skeleton:
    - hitung Euclidean DT di dalam foreground (fg_bool True)
    - ambil ridge = local maxima DT (neigh x neigh)
    - tipiskan jadi 1 piksel
    """
    dist = distance_transform_edt(fg_bool)  # radius ke background terdekat
    mx = maximum_filter(dist, size=neigh, mode='nearest')
    ridge = (dist == mx) & (dist > r_min) & fg_bool
    ridge_thin = thin(ridge)               # 1-pixel
    return ridge_thin.astype(np.uint8), dist


def _contour_metrics(mask_u8):
    """Return (area, perimeter, circularity, axis_ratio) for a binary mask (0/1 or 0/255)."""
    mu8 = ((mask_u8 > 0).astype(np.uint8) * 255)
    area = float(cv.countNonZero(mu8))
    if area <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cnts, _ = cv.findContours(mu8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        return area, 0.0, 0.0, 0.0
    c = max(cnts, key=cv.contourArea)
    per = float(cv.arcLength(c, True))
    circ = float((4.0 * math.pi * area) / (per * per + 1e-9))  # 1.0 = perfect circle

    axis_ratio = 0.0
    if len(c) >= 5:
        (cx, cy), (MA, ma), angle = cv.fitEllipse(c)
        major = float(max(MA, ma))
        minor = float(min(MA, ma))
        axis_ratio = float(minor / (major + 1e-9))              # 1.0 = circle
    return area, per, circ, axis_ratio


def carve_pseudo_holes_by_dt_peaks(
    fg_bool,
    peak_neigh=9,
    r_peak_min=4.0,
    alpha=0.78,
    circ_min=0.55,
    axis_min=0.55,
    max_carves=3,
    min_core_area=30,
    return_cores=False,
    debug=False
):

    """
    Deteksi gumpalan bulat lokal pada fg (tanpa hole) dengan DT peaks.
    Untuk tiap peak: buat core di sekitar peak, cek circularity, lalu "carve" (hapus core) -> pseudo-hole.
    Return:
      - default: (fg_new, carved_bool)
      - kalau return_cores=True: (fg_new, carved_bool, list_core_masks_bool)
    """
    fg = fg_bool.astype(bool)
    if fg.sum() == 0:
        return (fg, False, []) if return_cores else (fg, False)

    dist = distance_transform_edt(fg)
    mx = maximum_filter(dist, size=peak_neigh, mode="nearest")
    peaks = (dist == mx) & (dist >= r_peak_min) & fg

    ys, xs = np.where(peaks)
    if debug:
        print(f"[DT-PEAK] dist.max={dist.max():.3f} | peaks={len(xs)} | r_peak_min={r_peak_min}")

    if len(xs) == 0:
        return (fg, False, []) if return_cores else (fg, False)

    order = np.argsort(dist[ys, xs])[::-1]

    carved = False
    fg_new = fg.copy()
    carved_count = 0
    cores = []  # <-- TAMBAH: simpan core kandidat yang lolos

    for k in order:
        y0, x0 = int(ys[k]), int(xs[k])

        # update DT dari fg_new (karena fg_new berubah jika sudah carve)
        dist2 = distance_transform_edt(fg_new)
        r0 = float(dist2[y0, x0])
        if r0 < r_peak_min:
            continue

        core = (dist2 >= (alpha * r0)) & fg_new
        if core.sum() < min_core_area:
            continue

        # ambil hanya CC core yang mengandung peak
        core_u8 = (core.astype(np.uint8) * 255)
        nlab, lab = cv.connectedComponents((core_u8 > 0).astype(np.uint8), connectivity=8)
        if nlab <= 1:
            continue
        lab_id = lab[y0, x0]
        if lab_id == 0:
            continue

        core_cc_u8 = (lab == lab_id).astype(np.uint8) * 255

        # ukur circularity core_cc
        area, per, circ, axis = _contour_metrics(core_cc_u8)
        if area < min_core_area:
            continue
        if (circ < circ_min) and (axis < axis_min):
            continue

        # simpan mask core (sebelum di-carve)
        if return_cores:
            cores.append((core_cc_u8 > 0))  # bool mask

        # carve core -> pseudo-hole
        fg_new = fg_new & (core_cc_u8 == 0)
        carved = True
        carved_count += 1
        if carved_count >= max_carves:
            break

    if return_cores:
        return fg_new, carved, cores
    return fg_new, carved




def is_diacritic_component(area_px, cy, baseline_y, dot_area_max=350, dy_thresh=18):
    """Heuristik diakritik: kecil + jauh dari baseline teks (atas/bawah)."""
    return (area_px <= dot_area_max) and (abs(float(cy) - float(baseline_y)) > float(dy_thresh))

def create_circular_skeleton_from_dt(fg_bool, min_radius=2.0, dt_percentile=60):
    """
    Buat skeleton melingkar untuk gumpalan bulat.
    
    Strategi:
    1. Hitung DT (distance transform)
    2. Ambil threshold pada percentile tertentu dari DT
    3. Trace kontur dari area threshold ini
    4. Gabung dengan medial axis untuk hasil terbaik
    """
    fg = fg_bool.astype(bool)
    if fg.sum() == 0:
        return np.zeros_like(fg, dtype=np.uint8)
    
    # Hitung DT
    dist = distance_transform_edt(fg)
    
    # Cari threshold DT yang bagus (percentile-based)
    dist_vals = dist[fg]
    if len(dist_vals) == 0:
        return np.zeros_like(fg, dtype=np.uint8)
    
    # Ambil nilai DT pada percentile tertentu (misalnya 60% = area dalam)
    thresh_val = np.percentile(dist_vals, dt_percentile)
    thresh_val = max(thresh_val, min_radius)  # minimal min_radius
    
    # Area "dalam" gumpalan
    inner = (dist >= thresh_val) & fg
    
    if inner.sum() < 10:  # terlalu kecil
        # Fallback: medial axis + DT ridge
        skel_mat, _ = medial_axis(fg, return_distance=True)
        skel_dt, _ = dt_ridge_skeleton(fg, r_min=1.0, neigh=3)
        return thin(skel_mat | (skel_dt > 0)).astype(np.uint8)
    
    # --- Strategi 1: Kontur dari inner area ---
    inner_u8 = (inner.astype(np.uint8) * 255)
    
    # Erosi sedikit supaya kontur lebih ke dalam
    kernel_ero = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    inner_eroded = cv.erode(inner_u8, kernel_ero, iterations=1)
    
    contours, _ = cv.findContours(inner_eroded if np.any(inner_eroded) else inner_u8, 
                                   cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    skel_contour = np.zeros_like(fg, dtype=np.uint8)
    if len(contours) > 0:
        # Gambar kontur terbesar
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(skel_contour, [largest_contour], -1, 1, 1)
    
    # --- Strategi 2: Medial Axis (fallback) ---
    skel_mat, _ = medial_axis(fg, return_distance=True)
    
    # --- Strategi 3: DT Ridge ---
    skel_dt, _ = dt_ridge_skeleton(fg, r_min=1.0, neigh=3)
    
    # GABUNG SEMUA: kontur + medial + dt ridge
    skel_combined = (skel_contour > 0) | skel_mat | (skel_dt > 0)
    
    # Tipiskan jadi 1-pixel
    skel_final = thin(skel_combined).astype(np.uint8)
    
    return skel_final


def detect_blob_type(binary_mat, cy, baseline_y, median_body_h):
    """
    Klasifikasi komponen dengan threshold yang lebih longgar
    """
    area_px = cv.countNonZero(binary_mat)
    
    # 1) Hitung circularity
    area, per, circ, axis = _contour_metrics(binary_mat)
    
    # 2) Jarak dari baseline
    dy = abs(float(cy) - float(baseline_y))
    dy_thresh = max(12.0, 0.35 * median_body_h)
    
    # 3) DIAKRITIK: kecil + jauh dari baseline
    if area_px <= 350 and dy > dy_thresh:
        return 'diacritic', circ, axis, dy
    
    # 4) GUMPALAN BULAT: 
    # PERBAIKAN: threshold lebih longgar untuk catch lebih banyak kandidat
    if (300 <= area_px <= 3000 and      # range lebih lebar
        circ > 0.45 and                  # threshold lebih rendah (was 0.55)
        axis > 0.45 and                  # threshold lebih rendah (was 0.55)
        dy <= dy_thresh * 1.5):          # toleransi lebih besar (was 1.2)
        return 'circular_blob', circ, axis, dy
    
    # 5) Selain itu = huruf normal
    return 'normal', circ, axis, dy
        
def preserve_holes(fg255):
    """
    fg255: foreground putih (255), background hitam (0)
    Menghapus foreground yang masuk ke area hole (lubang) supaya loop (contoh wawu) tetap berlubang.
    """
    bg = cv.bitwise_not(fg255)  # background jadi putih
    h, w = bg.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    bg_ff = bg.copy()

    # banjiri background yang nyambung ke tepi (bukan hole)
    cv.floodFill(bg_ff, mask, (0, 0), 0)

    # yang tersisa putih (255) itu HOLE (lubang tertutup)
    holes = bg_ff  # 255 di hole, 0 di area lain

    # hapus foreground yang mengisi hole
    return cv.bitwise_and(fg255, cv.bitwise_not(holes))
    
def get_holes_mask(fg255):
    """
    fg255: binary dengan foreground=255 (huruf putih), background=0 (hitam)
    return: mask hole = 255 pada area lubang, 0 selain itu
    """
    fg255 = ((fg255 > 0).astype(np.uint8) * 255)

    # tambah border hitam supaya titik (0,0) pasti background
    pad = cv.copyMakeBorder(fg255, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)

    inv = cv.bitwise_not(pad)  # background jadi 255
    h, w = inv.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = inv.copy()

    # banjiri background yang nyambung ke tepi
    cv.floodFill(flood, mask, (0, 0), 0)

    # sisa 255 = hole (lubang tertutup)
    holes = flood[1:-1, 1:-1]  # buang padding
    return holes

def clean_noise_keep_text_band(bin255, row_frac=0.01, pad_y=35, border=12, min_area_inside=6):
    """
    1) Cari band teks dari proyeksi horizontal
    2) Nolkan (hapus total) area DI LUAR band teks -> noise atas/bawah langsung hilang
    3) Hapus komponen yang dekat border (termasuk blob kanan-atas)
    4) Hapus speckle super kecil di dalam band (tanpa bunuh titik yang normal)
    """
    bin01 = (bin255 > 0).astype(np.uint8)
    H, W = bin01.shape

    # --- cari band teks ---
    row_sum = bin01.sum(axis=1)
    thr = int(row_frac * W)  # mis 1% dari lebar
    ys = np.where(row_sum > thr)[0]

    if len(ys) > 0:
        y0 = max(0, int(ys.min()) - pad_y)
        y1 = min(H - 1, int(ys.max()) + pad_y)
    else:
        y0, y1 = 0, H - 1

    out = (bin01 * 255).copy()

    # --- hapus total area luar band (ini yang bunuh noise kanan-atas paling efektif) ---
    out[:y0, :] = 0
    out[y1+1:, :] = 0

    # --- connected components untuk bersih-bersih tambahan ---
    out01 = (out > 0).astype(np.uint8)
    n, labels, stats, cents = cv.connectedComponentsWithStats(out01, connectivity=8)

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        # buang yang dekat border (bukan cuma x==0)
        if x <= border or y <= border or (x + w) >= (W - border) or (y + h) >= (H - border):
            out[labels == i] = 0
            continue

        # buang speckle kecil di dalam band
        if area < min_area_inside:
            out[labels == i] = 0

    return out


def remove_noise_by_proximity(bin255, body_area=1200, body_h=26,
                              dot_area_max=900, dist_keep=20, border=12):
    """
    Hapus noise yang tidak dekat dengan badan huruf.
    - body_area/body_h: kriteria komponen "badan huruf" (stroke utama)
    - dot_area_max: komponen <= ini dianggap kandidat dot/noise (boleh cukup besar supaya noise blob kecil ikut masuk)
    - dist_keep: radius kedekatan (px) dari badan huruf untuk mempertahankan dot/diakritik
    - border: buang yang dekat pinggir
    """
    bw = (bin255 > 0).astype(np.uint8)
    H, W = bw.shape
    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)

    # 1) bentuk mask badan huruf (komponen besar / tinggi)
    body = np.zeros_like(bw)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= body_area or h >= body_h:
            body[labels == i] = 1

    # kalau body kosong (jaga-jaga), return apa adanya
    if body.sum() == 0:
        return bin255

    # 2) area "dekat badan huruf"
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*dist_keep+1, 2*dist_keep+1))
    near_body = cv.dilate(body, k, iterations=1)

    # 3) hapus kandidat noise kecil/menengah yang tidak menyentuh near_body
    out = (bw * 255).copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]

        # buang noise pinggir
        if x <= border or y <= border or (x + w) >= (W - border) or (y + h) >= (H - border):
            out[labels == i] = 0
            continue

        # hanya cek kandidat dot/noise (komponen kecil/menengah)
        if area <= dot_area_max:
            if np.count_nonzero(near_body[labels == i]) == 0:
                out[labels == i] = 0

    return out


def separate_connected_dots_aggressive(bin255, max_component_area=400, erosion_size=2):
    """
    Pisahkan dot yang terhubung dengan erosi agresif pada komponen kecil-menengah.
    Ini akan memisahkan dot yang nyambung ke dot lain atau ke huruf.
    
    Parameters:
    - max_component_area: komponen <= ini akan di-erosi
    - erosion_size: ukuran kernel erosi (makin besar makin agresif)
    """
    bw = (bin255 > 0).astype(np.uint8)
    H, W = bw.shape
    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)
    
    # Buat output dari binary asli
    out = bin255.copy()
    
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        
        # Hanya proses komponen kecil yang mungkin dot terhubung
        if area <= max_component_area:
            # Ambil komponen ini saja
            component_mask = (labels == i).astype(np.uint8) * 255
            
            # Erosi untuk memisahkan
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_size, erosion_size))
            eroded = cv.erode(component_mask, kernel, iterations=1)
            
            # Ganti area komponen asli dengan hasil erosi
            out[labels == i] = 0  # Hapus dulu
            out = cv.bitwise_or(out, eroded)  # Tambah hasil erosi
    
    return out




def remove_small_isolated_components(bin255, min_area=15, min_width=3, min_height=3):
    """
    Hapus komponen yang terlalu kecil (noise speckle).
    Ini untuk menghapus sisa-sisa dot kecil setelah erosi.
    
    Parameters:
    - min_area: area minimum untuk dipertahankan
    - min_width: lebar minimum
    - min_height: tinggi minimum
    """
    bw = (bin255 > 0).astype(np.uint8)
    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)
    
    out = (bw * 255).copy()
    
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        
        # Hapus komponen yang terlalu kecil
        if area < min_area or w < min_width or h < min_height:
            out[labels == i] = 0
    
    return out
        

def split_left_diacritic_pairs(gray255, left_frac=0.35, area_max=1200):
    """
    Pisahkan diakritik yang nyambung (bridge tipis) di sisi kiri gambar.
    Operasi murni binary: erode -> CC -> dilate-back per sub-komponen -> clip ke bentuk awal.

    gray255: binary 0/255
    left_frac: hanya proses komponen dengan x < left_frac*W (ujung kiri)
    area_max: hanya proses komponen kecil-menengah (diakritik/noise)
    """
    bw = (gray255 > 0).astype(np.uint8)
    H, W = bw.shape

    n, labels, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)
    out = (bw * 255).copy()

    # kandidat erosion paling ringan dulu
    candidates = [
        (cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)), 1),
        (cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), 1),
        (cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), 2),
    ]

    for i in range(1, n):
        x, y, w, h, area = stats[i]

        # fokus: komponen kecil di kiri (ujung kiri)
        if area > area_max:
            continue
        if x > int(left_frac * W):
            continue

        # ambil komponen ini saja
        comp = (labels == i).astype(np.uint8) * 255

        # coba erode sampai jembatan putus jadi >1 komponen
        best = None
        for ker, it in candidates:
            er = cv.erode(comp, ker, iterations=it)
            n2, lab2, st2, _ = cv.connectedComponentsWithStats((er > 0).astype(np.uint8), connectivity=8)

            if n2 - 1 >= 2:  # berhasil pecah jadi minimal 2 bagian
                best = (ker, it, er, lab2, st2, n2)
                break

        if best is None:
            continue  # tidak bisa dipecah dengan erosion ringan -> skip

        ker, it, er, lab2, st2, n2 = best

        # hapus komponen asli dulu dari output
        out[labels == i] = 0

        # reconstruct tiap sub-komponen: dilate balik + clip ke bentuk awal
        recon = np.zeros_like(comp)
        for j in range(1, n2):
            sub = (lab2 == j).astype(np.uint8) * 255
            dil = cv.dilate(sub, ker, iterations=it)
            dil = cv.bitwise_and(dil, comp)  # clip supaya gak melebar keluar bentuk awal
            recon = cv.bitwise_or(recon, dil)

        out = cv.bitwise_or(out, recon)

    return out




        
filename= sys.argv[1]
#filename= 'topanribut.png'
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
resz = cv.resize(image, (RESIZE_FACTOR*image.shape[1], RESIZE_FACTOR*image.shape[0]), interpolation=cv.INTER_LINEAR)
image= resz.copy()
image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 0) ENHANCEMENT (kontras huruf vs background)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(image_gray)
#img = cv.GaussianBlur(img, (3, 3), 0)

# Otsu untuk referensi ambang
T, _ = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 1) MASK HOLE (ketat) -> hanya buat ambil lubang wawu
BIAS_HOLE_UP = 35
Th = min(255, int(T) + BIAS_HOLE_UP)
_, th_hole = cv.threshold(img, Th, 255, cv.THRESH_BINARY)
holes0 = get_holes_mask(th_hole)

# 2) MASK HURUF UTAMA (lebih stabil untuk stroke pudar ujung)
# Adaptive membantu menangkap stroke pudar tanpa bikin blob global,
# lalu kita batasi lagi dengan Otsu-low supaya background nggak ikut putih.
th_adapt = cv.adaptiveThreshold(
    img, 255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    51,   # ganjil: 31/41/51
    -13    # kalau noise banyak: -9 / -11
)

OTSU_GATE_UP = 4   # coba 3–6
_, th_gate = cv.threshold(img, min(255, int(T) + OTSU_GATE_UP), 255, cv.THRESH_BINARY)
th_main = cv.bitwise_and(th_adapt, th_gate)


# 3) BRIDGE kecil (biar ujung tidak putus, tapi tidak nyambung antarhuruf)
kH = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))   # boleh 3 atau 5
th_main = cv.morphologyEx(th_main, cv.MORPH_CLOSE, kH, iterations=1)

# 4) DENOISE versi binary (ganti medianBlur)
kO = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
th_main = cv.morphologyEx(th_main, cv.MORPH_OPEN, kO, iterations=1)

# 5) BALIKIN HOLE (wawu tetap bulat)
th_main[holes0 == 255] = 0

gray = th_main

# ✅ Pisahkan diakritik nyambung di ujung kiri (binary-only)
gray = split_left_diacritic_pairs(gray, left_frac=0.35, area_max=1200)

# ✅ bersihin noise (terutama yang nempel di border)
gray = clean_noise_keep_text_band(gray, row_frac=0.01, pad_y=40, border=11, min_area_inside=6)

gray = separate_connected_dots_aggressive(gray, max_component_area=250, erosion_size=3)

# Tambahan: erosi khusus untuk dot vertikal
kernel_dot = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
bw_temp = (gray > 0).astype(np.uint8)
n, labels, stats, _ = cv.connectedComponentsWithStats(bw_temp, connectivity=8)
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 200 and h > w * 1.5:  # Komponen kecil yang vertikal
        component = (labels == i).astype(np.uint8) * 255
        eroded = cv.erode(component, kernel_dot, iterations=1)
        gray[labels == i] = 0
        gray = cv.bitwise_or(gray, eroded)
# ✅✅ TAMBAHAN: Hapus komponen kecil hasil sisa erosi
print("Menghapus noise kecil...")
gray = remove_small_isolated_components(gray, min_area=20, min_width=3, min_height=3)

# ✅✅ TAMBAHAN: Hapus dot yang jauh dari badan huruf
print("Menghapus dot yang tidak terhubung...")
gray = remove_noise_by_proximity(gray, body_area=1000, body_h=24, dot_area_max=600, dist_keep=22, border=12)

print("Otsu T =", T, "Th =", Th)






plt.figure(figsize=(10,3))
plt.imshow(gray, cmap='gray')
plt.title("Binary yang dipakai (gray)")
plt.axis('off')
plt.show()


#_, gray= cv.threshold(selective_eroded, 0, THREVAL, cv.THRESH_TRIANGLE) # works better with dynamic-selective erosion
#draw(gray)
render = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# Tampilkan grayscale
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')


_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
kernel = np.ones((2,2), np.uint8)

# 1) Opening → erosi, lalu dilasi → bantu buka lubang
opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)

# 2) Closing → dilasi, lalu eroai → jaga keutuhan huruf
closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=1)

# Visualisasi
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Thresholded (adaptive)")
plt.imshow(thresh, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Opened (2x2 kernel)")
plt.imshow(opened, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Opened + Closed")
plt.imshow(closed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()





#SLIC
cue = gray.copy()
slic = cv.ximgproc.createSuperpixelSLIC(cue,algorithm = cv.ximgproc.SLICO, region_size = SLIC_SPACE)
slic.iterate()
mask= slic.getLabelContourMask()
result_mask = cv.bitwise_and(cue, mask)
num_slic = slic.getNumberOfSuperpixels()
lbls = slic.getLabels()

# moments calculation for each superpixels, either voids or filled (in-stroke)
moments = [np.zeros((1, 2)) for _ in range(num_slic)]
moments_void = [np.zeros((1, 2)) for _ in range(num_slic)]
# tabulating the superpixel labels
for j in range(height):
    for i in range(width):
        if cue[j,i]!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)
            render[j,i,0]= 140-(10*(lbls[j,i]%6))
        else:
            moments_void[lbls[j,i]] = np.append(moments_void[lbls[j,i]], np.array([[i,j]]), axis=0)

#moments[0][1] = [0,0] # random irregularities, not quite sure why
# some badly needed 'sanity' check
def remove_zeros(moments):
    temp=[]
    v= len(moments)
    if v==1:
        return temp
    else:
        for p in range(v):
            if moments[p][0]!=0. and moments[p][1]!=0.:
                temp.append(moments[p])
        return temp

for n in range(len(moments)):
    moments[n]= remove_zeros(moments[n])

# draw(render)

######## // image preprocessing ends here

# generating nodes
scribe= nx.Graph() # start anew, just in case

# valid superpixel
filled=0
for n in range(num_slic):
    if ( len(moments[n])>SLIC_SPACE ): # remove spurious superpixel with area less than 2 px 
        cx= int( np.mean( [array[0] for array in moments[n]] )) # centroid
        cy= int( np.mean( [array[1] for array in moments[n]] ))
        if (cue[cy,cx]!=0):
            render[cy,cx,1] = 255 
            scribe.add_node(int(filled), label=int(lbls[cy,cx]), area=(len(moments[n])-1)/pow(SLIC_SPACE,2), hurf='', pos_bitmap=(cx,cy), pos_render=(cx,-cy), color='#FFA500', rasm=True)
            #print(f'point{n} at ({cx},{cy})')
            filled=filled+1

def pdistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# connected componentscv.circle(disp, pos[compodef line_iterator(img, point0, point1):
from dataclasses import dataclass, field
from typing import List
from typing import Optional

@dataclass
class ConnectedComponents:
    rect: (int,int,int,int) # from bounding rectangle
    centroid: (int,int) # centroid moment
    area: Optional[int] = field(default=0)
    nodes: List[int] = field(default_factory=list)
    mat: Optional[np.ndarray] = field(default=None, repr=False)
    node_start: Optional[int] = field(default=-1)    # right-up
    distance_start: Optional[int] = field(default=0) # right-up
    node_end: Optional[int] = field(default=-1)      # left-down
    distance_end: Optional[int] = field(default=0)   # left-down




pos = nx.get_node_attributes(scribe,'pos_bitmap')
components=[]
for n in range(scribe.number_of_nodes()):
    # fill
    seed= pos[n]
    ccv= gray.copy()
    cv.floodFill(ccv, None, seed, RASMVAL, loDiff=(3), upDiff=(3))
    _, ccv = cv.threshold(ccv, 100, RASMVAL, cv.THRESH_BINARY)
    mu= cv.moments(ccv)
    if mu['m00'] > pow(SLIC_SPACE,2)*PHI:
        mc= (int(mu['m10'] / (mu['m00'])), int(mu['m01'] / (mu['m00'])))
        area = mu ['m00']
        pd= pdistance(seed, mc)
        node_start = n
        box= cv.boundingRect(ccv)
        # append keypoint if the component already exists
        found=0
        for i in range(len(components)):
            if components[i].centroid==mc:
                components[i].nodes.append(n)
                # calculate the distance
                tvane= freeman(seed[0]-mc[0], mc[1]-seed[1] )
                #if seed[0]>mc[0] and pd>components[i].distance_start and (tvane==2 or tvane==4): # potential node_start for long rasm
                if seed[0]>mc[0] and pd>components[i].distance_start: # potential node_start
                    components[i].distance_start= pd
                    components[i].node_start= n
                elif seed[0]<mc[0] and pd>components[i].distance_end: # potential node_end
                    components[i].distance_end = pd
                    components[i].node_end= n
                found=1
                # print(f'old node[{n}] with component[{i}] at {mc} from {components[i].centroid} distance: {pd})')
                break
        if (found==0):
            components.append(ConnectedComponents(box, mc))
            idx= len(components)-1
            components[idx].nodes.append(n)
            components[idx].mat = ccv.copy()
            components[idx].area = int(mu['m00']/THREVAL)
            if seed[0]>mc[0]:
                components[idx].node_start= n
                components[idx].distance_start= pd
            else:
                components[idx].node_end= n
                components[idx].distance_end= pd
            #print(f'new node[{n}] with component[{idx}] at {mc} from {components[idx].centroid} distance: {pd})')


components = sorted(components, key=lambda x: x.centroid[0], reverse=True)
# for n in len(components):
#     for i in components[n].nodes:
#         distance= pdistance(components[n].centroid, pos[i])
#         print(f'{i}: {distance}')

# # drawing the starting node (bitmap level)
# disp = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
# for n in range(len(components)):
#     #print(f'{n} at {components[n].centroid} size {components[n].area}')
#     # draw green line for rasm at edges, color the rasm brighter
#     if components[n].area>4*PHI*pow(SLIC_SPACE,2):
#         disp= cv.bitwise_or(disp, cv.cvtColor(components[n].mat,cv.COLOR_GRAY2BGR))
#         seed= components[n].centroid
#         cv.circle(disp, seed, 2, (0,0,120), -1)
#         if components[n].node_start!=-1:
#             cv.circle(disp, pos[components[n].node_start], 2, (0,120,0), -1)
#         if components[n].node_end!=-1:
#             cv.circle(disp, pos[components[n].node_end], 2, (120,0,0), -1)
#         r= components[n].rect[0]+int(components[n].rect[2])
#         l= components[n].rect[0]
#         if l<width and r<width: # did we ever went beyond the frame?
#             for j1 in range(int(SLIC_SPACE*PHI),height-int(SLIC_SPACE*PHI)):
#                 disp[j1,r,1]= 120
#             for j1 in range(int(SLIC_SPACE*pow(PHI,3)),height-int(SLIC_SPACE*pow(PHI,3))):
#                 disp[j1,l,1]= 120
#     else:        
#         m= components[n].centroid[1]
#         i= components[n].centroid[0]
#         # draw blue line for shakil 'connection'
#         for j2 in range(int(m-(2*SLIC_SPACE*PHI)), int(m+(2*SLIC_SPACE*PHI))):
#             if j2<height and j2>0: 
#                 disp[j2,i,1]= RASMVAL/2
# draw(disp) 


# SKELETON (SimpleITK)
#skeleton_components = []

#for n in range(len(components)):
    #binary_mat = ((components[n].mat == RASMVAL).astype(np.uint8) * 255)

    #holes = get_holes_mask(binary_mat)   # jaga hole (wawu)
    #binary_mat[holes == 255] = 0

    #skeleton = sitk_thin_2d(binary_mat)  # <-- ganti ini saja
    #skeleton_components.append(skeleton)



# ====== SKELETON: HYBRID ONLY ======
skeleton_components_hyb = []

# --- Estimasi baseline teks (untuk bedakan huruf vs diakritik) ---
_body_ys = []
_body_hs = []
for _c in components:
    _x, _y, _w, _h = _c.rect
    _bin_tmp = ((_c.mat == RASMVAL).astype(np.uint8) * 255)
    _area_tmp = cv.countNonZero(_bin_tmp)
    if _area_tmp >= 1000 or _h >= 24:
        _body_ys.append(_c.centroid[1])
        _body_hs.append(_h)

if len(_body_ys) == 0:
    _body_ys = [c.centroid[1] for c in components]
    _body_hs = [c.rect[3] for c in components]

baseline_y = float(np.median(_body_ys)) if len(_body_ys) else 0.0
median_body_h = float(np.median(_body_hs)) if len(_body_hs) else 0.0
dy_thresh = max(12.0, 0.35 * median_body_h)

# ← PENTING: Simpan info tipe komponen
component_types = []

local_blob_circs = []   # circularity blob lokal (DT-core)
local_blob_axes  = []   # axis ratio blob lokal


# Pengecekan apakah ada komponen
if len(components) == 0:
    print("WARNING: Tidak ada komponen yang ditemukan untuk skeletonisasi!")
    combined_skeleton = np.zeros((height, width), dtype=np.uint8)
else:
    print(f"\n{'='*60}")
    print(f"MEMPROSES {len(components)} KOMPONEN")
    print(f"Baseline Y: {baseline_y:.1f}, Median Body Height: {median_body_h:.1f}")
    print(f"{'='*60}\n")
    
# ← LOOP PEMBUATAN SKELETON DENGAN VISUALISASI DEBUG
for n in range(len(components)):
    k_sk  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    k_dot = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))

    binary_mat = ((components[n].mat == RASMVAL).astype(np.uint8) * 255)
    holes = get_holes_mask(binary_mat)

    area_px = cv.countNonZero(binary_mat)
    cy = components[n].centroid[1]
    
    # ← DETEKSI TIPE KOMPONEN (dengan return nilai circ/axis)
    blob_type, circ, axis, dy = detect_blob_type(binary_mat, cy, baseline_y, median_body_h)
    component_types.append(blob_type)
    
    print(f"Komponen {n}: type={blob_type:15s} | area={area_px:4d} | "
          f"circ={circ:.3f} | axis={axis:.3f} | dy={dy:.1f}")
    
    # Morfologi sesuai ukuran
    if area_px >= 1200:
        binary_mat = cv.morphologyEx(binary_mat, cv.MORPH_CLOSE, k_sk, iterations=1)
    else:
        binary_mat = cv.morphologyEx(binary_mat, cv.MORPH_OPEN, k_dot, iterations=1)

    binary_mat[holes == 255] = 0
    fg = (binary_mat > 0)

    fg_used = fg
    forced_loop = False
    cores = []  # simpan core-core bulat lokal (DT-core) untuk komponen normal

    
    # ← STRATEGI SKELETON BERBEDA PER TIPE
    
    if blob_type == 'diacritic':
        # DIAKRITIK: skeleton sederhana
        skel_mat_bool, _ = medial_axis(fg_used, return_distance=True)
        skel_hyb = thin(skel_mat_bool).astype(np.uint8)
        print(f"  → Skeleton: Medial Axis (diakritik)")
        
    elif blob_type == 'circular_blob':
        # GUMPALAN BULAT: paksa skeleton melingkar!
        print(f"  → Skeleton: CIRCULAR (gumpalan bulat) *** SPECIAL TREATMENT ***")
        
        # Coba carve dulu
        if np.count_nonzero(holes) == 0:
            fg_used, forced_loop = carve_pseudo_holes_by_dt_peaks(
                fg,
                peak_neigh=7,        # lebih kecil untuk gumpalan kecil
                r_peak_min=3.0,      # lebih kecil
                alpha=0.75,
                circ_min=0.50,
                axis_min=0.50,
                max_carves=2,
                min_core_area=25
            )
            if forced_loop:
                print(f"  → Pseudo-hole berhasil di-carve!")
        
        # Buat skeleton melingkar dengan fungsi BARU
        skel_circular = create_circular_skeleton_from_dt(fg_used, min_radius=2.0, dt_percentile=60)
        
        # ← VISUALISASI PER KOMPONEN CIRCULAR BLOB
        if np.any(skel_circular > 0):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(binary_mat, cmap='gray')
            plt.title(f"Comp {n}: Binary Input")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(fg_used, cmap='gray')
            plt.title(f"FG after carve (carved={forced_loop})")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(skel_circular, cmap='gray')
            plt.title(f"Circular Skeleton (circ={circ:.2f})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        skel_hyb = skel_circular
        
    else:  # 'normal'

    # HURUF NORMAL: hybrid medial + DT + deteksi circular blob lokal (DT-core)

        fg_used, forced_loop, cores = carve_pseudo_holes_by_dt_peaks(
            fg,
            peak_neigh=7,          # lebih sensitif
            r_peak_min=2.2,        # TURUNKAN (ini paling ngaruh)
            alpha=0.70,            # lebih longgar
            circ_min=0.30,         # longgar dulu supaya kebaca
            axis_min=0.30,         # longgar dulu
            max_carves=8,
            min_core_area=18,
            return_cores=True,
            debug=True if n == 0 else False
        )
    
        # hitung circularity dari core-core yang lolos
        if len(cores) > 0:
            blob_circs = []
            blob_axes  = []
            for core_mask in cores:
                core_u8 = (core_mask.astype(np.uint8) * 255)
                a, p, c, ax = _contour_metrics(core_u8)
                blob_circs.append(c)
                blob_axes.append(ax)
    
            local_blob_circs.extend(blob_circs)
            local_blob_axes.extend(blob_axes)
    
            print(
                f"  → DT-core circular blobs (lokal): {len(cores)} | "
                f"circ avg/min/max = {np.mean(blob_circs):.3f}/"
                f"{np.min(blob_circs):.3f}/{np.max(blob_circs):.3f}"
            )
    
        # skeleton hybrid normal
        skel_mat_bool, _ = medial_axis(fg_used, return_distance=True)
        skel_dt, _ = dt_ridge_skeleton(fg_used, r_min=1.0, neigh=3)
        skel_hyb = thin((skel_mat_bool) | (skel_dt > 0)).astype(np.uint8)
    
        # paksa loop skeleton untuk core bulat lokal
        if len(cores) > 0:
            sk = (skel_hyb > 0)
            for core_mask in cores:
                sk_loop = (create_circular_skeleton_from_dt(core_mask, min_radius=2.0, dt_percentile=55) > 0)
                sk = sk | sk_loop
            skel_hyb = thin(sk).astype(np.uint8)
    
        print(f"  → Skeleton: Hybrid MAT+DT (normal)")

    
    skeleton_components_hyb.append(skel_hyb)

    print(f"\n{'='*60}\n")
    
    
# ← TAMBAHKAN INI SETELAH LOOP SELESAI
print(f"\n{'='*60}")
print("RINGKASAN DETEKSI TIPE KOMPONEN:")
print(f"{'='*60}")
print(f"Total komponen: {len(component_types)}")
print(f"- Diacritic:     {component_types.count('diacritic')}")
print(f"- Circular Blob: {component_types.count('circular_blob')}")  # ← CEK INI!
print(f"- Circular Blob (lokal DT-core): {len(local_blob_circs)}")
if len(local_blob_circs) > 0:
    print(f"  circ_local avg/min/max: {np.mean(local_blob_circs):.3f}/"
          f"{np.min(local_blob_circs):.3f}/{np.max(local_blob_circs):.3f}")

print(f"- Normal:        {component_types.count('normal')}")
print(f"{'='*60}\n")

# Gabungkan semua skeleton komponen
if len(skeleton_components_hyb) == 0:
    combined_skeleton = np.zeros((height, width), dtype=np.uint8)
else:
    combined_skeleton = np.zeros_like(skeleton_components_hyb[0], dtype=np.uint8)
    for s in skeleton_components_hyb:
        combined_skeleton |= (s > 0).astype(np.uint8)

# ← PINDAHKAN EDGE DETECTION KE SINI (SEBELUM EVALUASI)
# =============================================================
# DETEKSI TEPI (EDGE DETECTION)
# =============================================================
print("Menghitung edge detection...")
edges = cv.Canny(gray, 30, 30)

# Ambil kontur (outer + hole) dari hasil edge
contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Jadikan kontur menjadi gambar 1-pixel (0/255)
contour_img = np.zeros_like(edges, dtype=np.uint8)
cv.drawContours(contour_img, contours, -1, 255, 1)

# Distance Transform: jarak tiap piksel ke kontur terdekat
dt = cv.distanceTransform(cv.bitwise_not(contour_img), cv.DIST_L2, 5)

# =============================================================
# EVALUASI (SEKARANG edges SUDAH DIDEFINISIKAN)
# =============================================================
def evaluasi_skeleton_lengkap(skeleton, edge, component_types):
    """
    Evaluasi kualitas skeleton dengan circularity per komponen
    """
    panjang = np.sum(skeleton > 0)
    overlap = np.logical_and(skeleton, edge).sum()
    
    # Hitung connected components
    num_labels, labels = cv.connectedComponents(skeleton.astype(np.uint8))
    
    # Hitung circularity untuk tiap komponen skeleton
    circularities = []
    areas = []
    for i in range(1, num_labels):
        comp_mask = (labels == i).astype(np.uint8) * 255
        area, per, circ, axis = _contour_metrics(comp_mask)
        if area > 10:
            circularities.append(circ)
            areas.append(area)
    
    return {
        'panjang_skeleton': panjang,
        'jumlah_komponen_skeleton': num_labels - 1,
        'jumlah_komponen_input': len(component_types),
        'overlap_dengan_edge': overlap,
        'avg_circularity': np.mean(circularities) if len(circularities) > 0 else 0.0,
        'min_circularity': np.min(circularities) if len(circularities) > 0 else 0.0,
        'max_circularity': np.max(circularities) if len(circularities) > 0 else 0.0,
        'total_area_skeleton': np.sum(areas),
        'tipe_diacritic': component_types.count('diacritic'),
        'tipe_circular_blob': component_types.count('circular_blob'),
        'tipe_normal': component_types.count('normal'),
    }

# Visualisasi Skeleton
if combined_skeleton is not None and np.any(combined_skeleton > 0):
    plt.figure(figsize=(8, 6))
    plt.imshow(combined_skeleton, cmap='gray')
    plt.title("Hybrid Skeleton Medialaxis+Distance Transform+Circular Blob Detection")
    plt.axis('off')
    plt.show()
    
    # PANGGIL EVALUASI (edges sudah ada)
    hasil = evaluasi_skeleton_lengkap(combined_skeleton, edges, component_types)
    print("\n" + "="*60)
    print("EVALUASI SKELETON")
    print("="*60)
    for k, v in hasil.items():
        if isinstance(v, float):
            print(f"{k:30s}: {v:.4f}")
        else:
            print(f"{k:30s}: {v}")
    print("="*60 + "\n")
else:
    print("Skeleton kosong, tidak ada data untuk ditampilkan.")
    
    


# =============================================================
# VISUALISASI TAMBAHAN: Edge, Contour, DT Map
# =============================================================
print("Menampilkan visualisasi edge detection...")

# Ambil jarak hanya di titik skeleton
if combined_skeleton is not None and np.any(combined_skeleton > 0):
    dist_on_skel = dt[combined_skeleton > 0]

    if dist_on_skel.size > 0:
        print("DT(skeleton->contour) mean/median/min/max:",
              float(dist_on_skel.mean()),
              float(np.median(dist_on_skel)),
              float(dist_on_skel.min()),
              float(dist_on_skel.max()))
    else:
        print("Skeleton kosong / tidak overlap dengan dt.")


# DETEKSI TEPI (EDGE DETECTION)
edges = cv.Canny(gray, 30, 30)

# Ambil kontur (outer + hole) dari hasil edge
contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Jadikan kontur menjadi gambar 1-pixel (0/255)
contour_img = np.zeros_like(edges, dtype=np.uint8)
cv.drawContours(contour_img, contours, -1, 255, 1)

# Distance Transform: jarak tiap piksel ke kontur terdekat
# distanceTransform menghitung jarak ke piksel bernilai 0 terdekat,
# jadi kita invert: kontur(255)->0, selainnya(0)->255
dt = cv.distanceTransform(cv.bitwise_not(contour_img), cv.DIST_L2, 5)  # float32

# Ambil jarak hanya di titik skeleton
if combined_skeleton is not None:
    dist_on_skel = dt[combined_skeleton > 0]

    if dist_on_skel.size > 0:
        print("DT(skeleton->contour) mean/median/min/max:",
              float(dist_on_skel.mean()),
              float(np.median(dist_on_skel)),
              float(dist_on_skel.min()),
              float(dist_on_skel.max()))
    else:
        print("Skeleton kosong / tidak overlap dengan dt.")
else:
    print("combined_skeleton = None (tidak ada skeleton).")

# Opsional: tampilkan hasil deteksi tepi / kontur untuk verifikasi
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.show()

plt.imshow(contour_img, cmap="gray")
plt.title("Contour Image (1px)")
plt.show()


plt.figure(figsize=(10,3))
plt.imshow(dt, cmap='jet')          # peta jarak
plt.imshow(combined_skeleton>0, cmap='gray', alpha=0.6)  # skeleton overlay
plt.title("DT map + skeleton overlay")
plt.axis('off')
plt.show()


# def evaluasi_skeleton(skeleton, edge):
#     panjang = np.sum(skeleton > 0)
#     num_labels, _ = cv.connectedComponents(skeleton.astype(np.uint8))
#     overlap = np.logical_and(skeleton, edge).sum()
#     return {
#         'panjang_skeleton': panjang,
#         # 'jumlah_komponen': num_labels - 1,
#         'overlap_dengan_edge': overlap
#     }

# hasil = evaluasi_skeleton(combined_skeleton, edges)
# for k, v in hasil.items():
#     print(f"{k}: {v}")
