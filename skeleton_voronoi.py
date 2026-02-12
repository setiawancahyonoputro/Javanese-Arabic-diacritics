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
from scipy.spatial import Voronoi
from skimage.morphology import skeletonize, medial_axis


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

def vma_skeleton(binary255,
                 sample_step=3,
                 prune_dt_ratio=0.18,
                 chord_min_px=4.0,
                 chord_ratio_min=0.6,
                 do_thin=True):
    """
    VMA eksplisit:
    - Ambil boundary points (outer + holes) dari binary foreground
    - Voronoi(sites)
    - Ambil ridge/edge Voronoi yang vertex-nya berada DI DALAM shape
    - Rasterize edge -> skeleton
    - Prune spurs pakai DT threshold (radius kecil dibuang)
    - (opsional) prune pakai "chord residual" proxy:
        chord = jarak 2 site boundary yg membentuk ridge
        r = radius (dist vertex ke site)
        buang jika chord terlalu kecil relatif ke r

    binary255: foreground=255, background=0
    return: skeleton uint8 {0,1}
    """
    bw = (binary255 > 0).astype(np.uint8)
    H, W = bw.shape
    if bw.sum() == 0:
        return np.zeros_like(bw, dtype=np.uint8)

    # boundary points (outer + holes)
    contours, _ = cv.findContours(bw, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros_like(bw, dtype=np.uint8)

    pts = []
    for c in contours:
        c = c.reshape(-1, 2)
        if sample_step > 1:
            c = c[::sample_step]
        if len(c) > 0:
            pts.append(c)

    if not pts:
        return np.zeros_like(bw, dtype=np.uint8)

    sites = np.vstack(pts).astype(np.float64)
    sites = np.unique(sites, axis=0)  # buang duplikat

    # terlalu sedikit point -> fallback
    if len(sites) < 10:
        return skeletonize(bw > 0).astype(np.uint8)

    # Voronoi
    try:
        vor = Voronoi(sites)
    except Exception:
        return skeletonize(bw > 0).astype(np.uint8)

    V = vor.vertices  # (Nv,2) float
    if V is None or len(V) == 0:
        return skeletonize(bw > 0).astype(np.uint8)

    # cek vertex Voronoi yang jatuh di dalam shape
    Vi = np.rint(V).astype(int)
    ok = (Vi[:, 0] >= 0) & (Vi[:, 0] < W) & (Vi[:, 1] >= 0) & (Vi[:, 1] < H)
    inside = np.zeros(len(V), dtype=bool)
    idx_ok = np.where(ok)[0]
    inside[idx_ok] = (bw[Vi[idx_ok, 1], Vi[idx_ok, 0]] > 0)

    # DT-inside = radius ke background (boundary)
    dt_in = cv.distanceTransform((bw * 255).astype(np.uint8), cv.DIST_L2, 5)
    rmax = float(dt_in.max()) if dt_in.size else 0.0
    dt_thr = prune_dt_ratio * rmax if rmax > 0 else 0.0

    sk = np.zeros_like(bw, dtype=np.uint8)

    # Rasterize ridge Voronoi yang valid
    for (p_idx, q_idx), (v0, v1) in zip(vor.ridge_points, vor.ridge_vertices):
        if v0 == -1 or v1 == -1:
            continue
        if not (inside[v0] and inside[v1]):
            continue

        # --- chord residual proxy pruning ---
        p = sites[p_idx]
        q = sites[q_idx]
        chord = float(np.hypot(p[0] - q[0], p[1] - q[1]))
        if chord < chord_min_px:
            continue

        # radius di endpoint-edge (pakai site p sebagai referensi)
        r0 = float(np.hypot(V[v0, 0] - p[0], V[v0, 1] - p[1]))
        r1 = float(np.hypot(V[v1, 0] - p[0], V[v1, 1] - p[1]))
        r = min(r0, r1)
        if r > 1e-6 and (chord / (2.0 * r) < chord_ratio_min):
            continue
        # ------------------------------------

        x0, y0 = int(round(V[v0, 0])), int(round(V[v0, 1]))
        x1, y1 = int(round(V[v1, 0])), int(round(V[v1, 1]))
        cv.line(sk, (x0, y0), (x1, y1), 1, 1)

    # pastikan skeleton tetap di area foreground
    sk = (sk & bw).astype(np.uint8)

    # prune spurs dekat boundary pakai DT threshold
    if dt_thr > 0 and sk.sum() > 0:
        sk = ((sk > 0) & (dt_in >= dt_thr)).astype(np.uint8)

    # thinning final biar 1-pixel rapi
    if do_thin and sk.sum() > 0:
        sk = skeletonize(sk > 0).astype(np.uint8)

    return sk


        
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



# SKELETON OLD
skeleton_components = []
for n in range(len(components)):
    k_sk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    k_dot = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    binary_mat = ((components[n].mat == RASMVAL).astype(np.uint8) * 255)
    
    # simpan hole per komponen
    holes = get_holes_mask(binary_mat)
    
    # hitung ukuran komponen (pixel putih)
    area_px = cv.countNonZero(binary_mat)
    
    # CLOSE hanya untuk komponen besar (huruf), bukan dot
    if area_px >= 1200:
        binary_mat = cv.morphologyEx(binary_mat, cv.MORPH_CLOSE, k_sk, iterations=1)
    else:
        binary_mat = cv.morphologyEx(binary_mat, cv.MORPH_OPEN, k_dot, iterations=1)
    
    binary_mat[holes == 255] = 0
    
    skeleton = vma_skeleton(
    binary_mat,
    sample_step=3,
    prune_dt_ratio=0.15,
    chord_min_px=1.5,
    chord_ratio_min=0.4,
    do_thin=True
).astype(np.uint8)

    skeleton_components.append(skeleton)
    
    
    
# Gabungkan semua skeleton menjadi satu gambar
if len(skeleton_components) > 0:
    combined_skeleton = np.zeros_like(skeleton_components[0], dtype=np.uint8)

    for skeleton in skeleton_components:
        combined_skeleton |= skeleton
else:
    combined_skeleton = None

# Periksa apakah skeleton kosong sebelum menampilkan
if combined_skeleton is not None and np.any(combined_skeleton > 0):
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_skeleton, cmap='gray')
    plt.title("Skeletonizationvoromoi")
    plt.axis('off')
    plt.show()
else:
    print("Skeleton kosong, tidak ada data untuk ditampilkan.")
    

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
