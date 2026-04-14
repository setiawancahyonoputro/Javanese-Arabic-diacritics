# SCRIPT BERHASIL MEMOTONG HURUF SESUAI DENGAN STUKTUR HURUFNYA PER SUB-PATH
# SKELETON: ZHANG-SUEN THINNING (Zhang & Suen, 1984)
# Referensi: T.Y. Zhang and C.Y. Suen, "A Fast Parallel Algorithm for Thinning Digital Patterns",
#             Communications of the ACM, 27(3), pp. 236–239, 1984.

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import math
from skimage.morphology import skeletonize, thin
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
from skimage.filters import threshold_local
# medial_axis dihapus — skeletonisasi murni Zhang-Suen (T.Y. Zhang & C.Y. Suen, 1984)
from scipy.ndimage import distance_transform_edt, maximum_filter
import csv
from collections import defaultdict, Counter

PHI= 1.6180339887498948482 # ppl says this is a beautiful number :)

# ============================================================
# PARAMETER UTAMA: THRESHOLD DT UNTUK DETEKSI LOOP
# ============================================================
DT_LOOP_THRESHOLD = 5.0   # ubah sesuai kebutuhan (px)


# ============================================================
# ZHANG-SUEN THINNING
# Implementasi algoritma Zhang-Suen (1984) untuk skeletonisasi.
# Input : fg_bool (bool ndarray) — foreground True, background False
# Output: skeleton (uint8 ndarray) — 0/1, skeleton 1-pixel
# ============================================================
def zhang_suen_thinning(fg_bool):
    """
    Zhang-Suen Thinning Algorithm (1984).

    Referensi:
        T.Y. Zhang and C.Y. Suen,
        "A Fast Parallel Algorithm for Thinning Digital Patterns",
        Communications of the ACM, 27(3), pp. 236-239, 1984.

    Bekerja dalam dua sub-iterasi per iterasi:
      Sub-iter 1: hapus piksel dengan kondisi tertentu pada tetangga 8-konektivitas
      Sub-iter 2: hapus piksel dengan kondisi cerminan sub-iter 1
    Berlanjut sampai tidak ada piksel yang berubah (konvergen).

    Parameters
    ----------
    fg_bool : np.ndarray (H, W) bool
        Foreground True, background False

    Returns
    -------
    skel : np.ndarray (H, W) uint8
        Skeleton Zhang-Suen, nilai 0 atau 1
    """
    img = fg_bool.astype(np.uint8)
    prev = np.zeros_like(img)

    def _zhang_suen_step(img, step):
        """
        Satu sub-iterasi Zhang-Suen.
        step=1 → sub-iterasi 1
        step=2 → sub-iterasi 2
        Return: mask piksel yang akan DIHAPUS
        """
        # Ambil 8 tetangga:
        # P2 P3 P4
        # P9  P  P5
        # P8 P7 P6
        P2 = np.roll(img, -1, axis=0)        # atas
        P3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)   # kanan-atas
        P4 = np.roll(img,  1, axis=1)         # kanan
        P5 = np.roll(np.roll(img,  1, axis=0), 1, axis=1)   # kanan-bawah
        P6 = np.roll(img,  1, axis=0)         # bawah
        P7 = np.roll(np.roll(img,  1, axis=0), -1, axis=1)  # kiri-bawah
        P8 = np.roll(img, -1, axis=1)         # kiri
        P9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)  # kiri-atas

        # B(P) = jumlah tetangga yang bernilai 1
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

        # A(P) = jumlah transisi 0→1 dalam urutan melingkar P2,P3,...,P9,P2
        neighbors_seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
        A = sum(
            ((neighbors_seq[i] == 0) & (neighbors_seq[i+1] == 1)).astype(np.uint8)
            for i in range(8)
        )

        # Kondisi umum (sama untuk kedua sub-iter):
        cond1 = (img == 1)           # piksel aktif
        cond2 = (B >= 2) & (B <= 6) # jumlah tetangga antara 2–6
        cond3 = (A == 1)             # tepat 1 transisi 0→1

        # Kondisi khusus per sub-iterasi:
        if step == 1:
            # Sub-iter 1: P2*P4*P6==0 dan P4*P6*P8==0
            cond4 = (P2 * P4 * P6 == 0)
            cond5 = (P4 * P6 * P8 == 0)
        else:
            # Sub-iter 2: P2*P4*P8==0 dan P2*P6*P8==0
            cond4 = (P2 * P4 * P8 == 0)
            cond5 = (P2 * P6 * P8 == 0)

        return cond1 & cond2 & cond3 & cond4 & cond5

    while True:
        # Sub-iterasi 1 (Zhang-Suen)
        del1 = _zhang_suen_step(img, step=1)
        img[del1] = 0

        # Sub-iterasi 2 (Zhang-Suen)
        del2 = _zhang_suen_step(img, step=2)
        img[del2] = 0

        # Konvergensi: tidak ada perubahan
        if not np.any(del1) and not np.any(del2):
            break

    return img.astype(np.uint8)


def freeman(x, y):
    if (y==0):
        y=1e-9
    if (x==0):
        x=-1e-9
    if (abs(x/y)<pow(PHI,2)) and (abs(y/x)<pow(PHI,2)):
        if   (x>0) and (y>0):
            return(1)
        elif (x<0) and (y>0):
            return(3)
        elif (x<0) and (y<0):
            return(5)
        elif (x>0) and (y<0):
            return(7)
    else:
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

# ============================================================
# MODE CLEANING BINARY
# USE_GATE_OTSU_BINARY  -> output binary mengikuti panel "Gate Otsu longgar"
#                          diakritik ikut dipertahankan
# USE_BODY_MASK_BINARY  -> output binary mengikuti body mask (bersih),
#                          tetapi titik/diakritik kecil yang terpisah bisa hilang
# Jika keduanya False   -> pakai jalur despeckle + rescue yang lebih konservatif
# Prioritas mode        -> GATE_OTSU > BODY_MASK > CONSERVATIVE
# ============================================================
USE_GATE_OTSU_BINARY = True
USE_BODY_MASK_BINARY = False
BODY_MASK_AREA = 220
BODY_MASK_H = 18
BODY_MASK_W_FRAC = 0.035
BODY_MASK_ROW_FRAC = 0.002
BODY_MASK_PAD_Y = 55
BODY_MASK_CLOSE_ITER = 0



def draw(img):
    plt.figure(dpi=600)
    plt.grid(False)
    if (len(img.shape)==3):
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    elif (len(img.shape)==2):
        plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))


def dt_ridge_skeleton_zhang(fg_bool, r_min=1.0, neigh=3):
    """
    DT-based skeleton menggunakan Zhang-Suen thinning sebagai pengganti thin().

    1. Hitung Euclidean DT di dalam foreground
    2. Ambil ridge = local maxima DT
    3. Tipiskan dengan Zhang-Suen thinning (1984)
    """
    dist = distance_transform_edt(fg_bool)
    mx = maximum_filter(dist, size=neigh, mode='nearest')
    ridge = (dist == mx) & (dist > r_min) & fg_bool
    # ← ZHANG: thinning via Zhang-Suen
    ridge_thin = zhang_suen_thinning(ridge)
    return ridge_thin.astype(np.uint8), dist


def contour_dt_to_edge(bin255):
    bin255 = ((bin255 > 0).astype(np.uint8) * 255)
    contours, hier = cv.findContours(bin255, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contour_img = np.zeros_like(bin255, dtype=np.uint8)
    if len(contours) > 0:
        cv.drawContours(contour_img, contours, -1, 255, 1)
    src = np.full_like(contour_img, 255, dtype=np.uint8)
    src[contour_img > 0] = 0
    dt_edge = cv.distanceTransform(src, cv.DIST_L2, 5)
    return contour_img, dt_edge


def dt_map_at_skeleton(skel_u8, dt_edge):
    sk = (skel_u8 > 0)
    dt_skel_map = np.zeros_like(dt_edge, dtype=np.float32)
    dt_skel_map[sk] = dt_edge[sk]
    return dt_skel_map, dt_edge[sk]


def _contour_metrics(mask_u8):
    mu8 = ((mask_u8 > 0).astype(np.uint8) * 255)
    area = float(cv.countNonZero(mu8))
    if area <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cnts, _ = cv.findContours(mu8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        return area, 0.0, 0.0, 0.0
    c = max(cnts, key=cv.contourArea)
    per = float(cv.arcLength(c, True))
    circ = float((4.0 * math.pi * area) / (per * per + 1e-9))
    axis_ratio = 0.0
    if len(c) >= 5:
        (cx, cy), (MA, ma), angle = cv.fitEllipse(c)
        major = float(max(MA, ma))
        minor = float(min(MA, ma))
        axis_ratio = float(minor / (major + 1e-9))
    return area, per, circ, axis_ratio


def _bbox_extent_roundness(mask_u8):
    mu8 = ((mask_u8 > 0).astype(np.uint8) * 255)
    cnts, _ = cv.findContours(mu8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0, 0, 0
    c = max(cnts, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)
    bbox_ratio = float(min(w, h) / (max(w, h) + 1e-9))
    extent = float(cv.contourArea(c) / (float(w * h) + 1e-9))
    return bbox_ratio, extent, int(w), int(h)


def _skeleton_topology_stats(skel_u8):
    """
    Statistik topologi sederhana untuk skeleton 1px:
    - skel_len   : jumlah piksel skeleton
    - endpoints  : jumlah titik ujung (degree==1)
    - branches   : jumlah titik cabang (degree>=3)
    """
    sk = (skel_u8 > 0).astype(np.uint8)
    ys, xs = np.where(sk > 0)
    coords = list(zip(xs.tolist(), ys.tolist()))
    if len(coords) == 0:
        return 0, 0, 0

    coord_set = set(coords)
    endpoints = 0
    branches = 0
    for (x, y) in coords:
        deg = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in coord_set:
                    deg += 1
        if deg == 1:
            endpoints += 1
        elif deg >= 3:
            branches += 1

    return int(len(coords)), int(endpoints), int(branches)


def _is_elongated_open_stroke(mask_u8, skel_u8=None,
                              bbox_ratio_max=0.42,
                              aspect_min=2.4,
                              circ_max=0.32,
                              extent_max=0.22,
                              min_skel_pixels=10,
                              skel_len_ratio_min=0.85):
    """
    Deteksi guard untuk stroke tebal memanjang yang SEHARUSNYA tetap
    diskeletonisasi normal, bukan diubah menjadi loop / contour luar.

    Intuisi:
    - bentuknya memanjang / tidak kompak
    - skeleton lokalnya berupa path terbuka (punya endpoint), bukan loop tertutup
    """
    mu8 = ((mask_u8 > 0).astype(np.uint8) * 255)
    area = cv.countNonZero(mu8)
    if area <= 0:
        return False

    _area_tmp, _per_tmp, circ, axis = _contour_metrics(mu8)
    bbox_ratio, extent, bw, bh = _bbox_extent_roundness(mu8)
    aspect = float(max(bw, bh) / (min(bw, bh) + 1e-9))

    if skel_u8 is None:
        skel_u8 = zhang_suen_thinning(mu8 > 0).astype(np.uint8)
    skel_len, endpoints, branches = _skeleton_topology_stats(skel_u8)

    elongated_shape = (
        (aspect >= aspect_min) or
        (bbox_ratio <= bbox_ratio_max) or
        ((circ <= circ_max) and (extent <= extent_max))
    )

    open_path_like = (
        (endpoints >= 2) and
        (branches <= 1) and
        (skel_len >= max(min_skel_pixels, int(skel_len_ratio_min * max(bw, bh))))
    )

    return bool(elongated_shape and open_path_like)


def _is_compact_loop_candidate(mask_u8, skel_u8=None,
                               circ_min=0.32,
                               round_min=0.48,
                               extent_min=0.16,
                               aspect_max=2.4,
                               min_area=12):
    """
    Kandidat loop lokal yang BOLEH dikembangkan menjadi contour-loop.
    Region tebal yang memanjang / berupa stroke terbuka otomatis ditolak.
    """
    mu8 = ((mask_u8 > 0).astype(np.uint8) * 255)
    area = cv.countNonZero(mu8)
    if area < min_area:
        return False

    if _is_elongated_open_stroke(mu8, skel_u8=skel_u8):
        return False

    _area_tmp, _per_tmp, circ, axis = _contour_metrics(mu8)
    bbox_ratio, extent, bw, bh = _bbox_extent_roundness(mu8)
    aspect = float(max(bw, bh) / (min(bw, bh) + 1e-9))
    roundness = float(max(axis, bbox_ratio))

    if circ < circ_min:
        return False
    if roundness < round_min:
        return False
    if extent < extent_min:
        return False
    if aspect > aspect_max:
        return False

    return True


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
    debug=False,
    extent_min=0.30,
    bbox_min=0.45,
    core_ratio_max=0.65,
    max_len_vs_r0=6.0
):
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

    all_r0 = dist[ys, xs]
    r0_median = float(np.median(all_r0))
    r0_blob_thresh = r0_median * 1.15
    if debug:
        print(f"[DT-PEAK] r0 median={r0_median:.2f} | blob_thresh={r0_blob_thresh:.2f} | "
              f"r0 min/max={all_r0.min():.2f}/{all_r0.max():.2f}")

    order = np.argsort(all_r0)[::-1]
    carved = False
    fg_new = fg.copy()
    carved_count = 0
    cores = []

    for k in order:
        y0, x0 = int(ys[k]), int(xs[k])
        dist2 = distance_transform_edt(fg_new)
        r0 = float(dist2[y0, x0])

        if r0 < r_peak_min:
            continue
        if r0 < r0_blob_thresh:
            break

        _ring_r_inner = r0 * 0.60
        _ring_r_outer = r0 * 0.90
        _Y, _X = np.ogrid[:dist2.shape[0], :dist2.shape[1]]
        _d_from_peak = np.sqrt((_Y - y0)**2 + (_X - x0)**2)
        _ring_mask = (_d_from_peak >= _ring_r_inner) & (_d_from_peak <= _ring_r_outer) & fg_new
        if _ring_mask.sum() > 0:
            _r0_ring_mean = float(dist2[_ring_mask].mean())
            if _r0_ring_mean > r0 * 0.80:
                continue

        # ← ZHANG: skeletonize menggunakan Zhang-Suen
        _skel_tmp = zhang_suen_thinning(fg_new).astype(np.uint8)
        _peak_r = max(3, int(r0 * 0.3))
        _Y2, _X2 = np.ogrid[:fg_new.shape[0], :fg_new.shape[1]]
        _d_peak = np.sqrt((_Y2 - y0)**2 + (_X2 - x0)**2)
        _ring_skel = _skel_tmp & (_d_peak > _peak_r) & (_d_peak <= _peak_r * 2.5)
        _n_comp_ring, _ = cv.connectedComponents(_ring_skel.astype(np.uint8), connectivity=8)
        _n_branches = _n_comp_ring - 1
        if _n_branches >= 3:
            if debug:
                print(f"  [JUNCTION SKIP] r0={r0:.2f} branches_out={_n_branches}")
            continue

        core = (dist2 >= (alpha * r0)) & fg_new
        if core.sum() < min_core_area:
            continue

        core_u8 = (core.astype(np.uint8) * 255)
        nlab, lab = cv.connectedComponents((core_u8 > 0).astype(np.uint8), connectivity=8)
        if nlab <= 1:
            continue
        lab_id = lab[y0, x0]
        if lab_id == 0:
            continue

        core_cc_u8 = (lab == lab_id).astype(np.uint8) * 255
        area, per, circ, axis = _contour_metrics(core_cc_u8)
        if area < min_core_area:
            continue

        bbox_ratio, extent, bw, bh = _bbox_extent_roundness(core_cc_u8)
        roundness = float(max(axis, bbox_ratio))
        bbox_max = float(max(bw, bh))

        if (circ < circ_min) or (roundness < axis_min):
            continue
        if extent < extent_min:
            continue
        if bbox_ratio < 0.45:
            continue
        if bw > 0 and bh > 0 and float(max(bw,bh)) / float(min(bw,bh)+1e-9) > 2.2:
            continue

        core_ratio = float(area / (math.pi * (r0 * r0 + 1e-9)))
        if core_ratio > core_ratio_max:
            continue
        if bbox_max > (max_len_vs_r0 * r0):
            continue

        if debug:
            print(f"  [CORE ACCEPTED] r0={r0:.2f} circ={circ:.2f} axis={axis:.2f} "
                  f"extent={extent:.2f} core_ratio={core_ratio:.2f} bbox_max={bbox_max:.1f}")

        if return_cores:
            cores.append((core_cc_u8 > 0))

        fg_new = fg_new & (core_cc_u8 == 0)
        carved = True
        carved_count += 1
        if carved_count >= max_carves:
            break

    if return_cores:
        return fg_new, carved, cores
    return fg_new, carved


def develop_loop_from_threshold(fg_bool, skel_u8, dt_loop_threshold=5.0):
    """
    Pada lokasi tebal (DT tinggi), loop hanya dikembangkan untuk region yang
    cukup kompak / blob-like.

    Region tebal yang sebenarnya hanyalah STROKE MEMANJANG TERBUKA
    tidak boleh diubah menjadi contour-loop. Region seperti itu dibiarkan
    memakai skeleton normal yang sudah ada.
    """
    from scipy.ndimage import distance_transform_edt as edt_scipy

    fg = fg_bool.astype(bool)
    sk = (skel_u8 > 0).astype(np.uint8)

    if fg.sum() == 0 or sk.sum() == 0:
        return sk, np.zeros_like(sk), np.zeros(fg.shape, dtype=bool)

    dist = edt_scipy(fg)
    sk_thick = sk & (dist > dt_loop_threshold)

    if sk_thick.sum() == 0:
        return sk, np.zeros_like(sk), np.zeros(fg.shape, dtype=bool)

    loop_mask = np.zeros_like(sk, dtype=np.uint8)
    region_thick_all = (dist > dt_loop_threshold) & fg
    replaced_region = np.zeros(fg.shape, dtype=bool)

    if region_thick_all.sum() == 0:
        return sk, loop_mask, replaced_region

    n_thick, labels_thick = cv.connectedComponents(
        region_thick_all.astype(np.uint8), connectivity=8
    )

    for lab_id in range(1, n_thick):
        blob_mask = (labels_thick == lab_id)
        if not np.any(sk_thick & blob_mask):
            continue

        fg_local = fg & blob_mask
        if fg_local.sum() < 5:
            continue

        fg_local_u8 = (fg_local.astype(np.uint8) * 255)
        sk_local_u8 = ((sk & blob_mask) > 0).astype(np.uint8)

        # Guard utama:
        # jika region tebal ini sebenarnya stroke memanjang terbuka,
        # JANGAN diganti contour loop.
        if not _is_compact_loop_candidate(fg_local_u8, skel_u8=sk_local_u8):
            continue

        contours_local, _ = cv.findContours(
            fg_local_u8, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE
        )
        if len(contours_local) == 0:
            continue

        loop_local = np.zeros_like(sk, dtype=np.uint8)
        cv.drawContours(loop_local, contours_local, -1, 1, 1)

        # ← ZHANG: thinning loop menggunakan Zhang-Suen
        loop_thin = zhang_suen_thinning(loop_local > 0).astype(np.uint8)
        if np.any(loop_thin > 0):
            loop_mask |= loop_thin
            replaced_region |= blob_mask

    # Hanya region yang BENAR-BENAR diganti loop saja yang dipotong dari skeleton awal.
    # Region tebal memanjang yang di-skip tetap mempertahankan skeleton normalnya.
    sk_normal = sk & (~replaced_region)

    if np.any(loop_mask > 0):
        skel_result = zhang_suen_thinning(
            (sk_normal > 0) | (loop_mask > 0)
        ).astype(np.uint8)
    else:
        skel_result = sk.copy().astype(np.uint8)

    return skel_result, loop_mask, replaced_region


def is_diacritic_component(area_px, cy, baseline_y, dot_area_max=350, dy_thresh=18):
    return (area_px <= dot_area_max) and (abs(float(cy) - float(baseline_y)) > float(dy_thresh))


def create_circular_skeleton_zhang(fg_bool, min_radius=2.0, dt_percentile=60, force_circular=False):
    """
    Skeleton melingkar untuk gumpalan bulat.

    Untuk circular blob, targetnya adalah loop / cincin kosong di tengah,
    jadi fallback TIDAK boleh kembali ke skeleton Zhang penuh di bagian dalam blob.
    """
    fg = fg_bool.astype(bool)
    if fg.sum() == 0:
        return np.zeros_like(fg, dtype=np.uint8)

    _br, _ext, _bw, _bh = _bbox_extent_roundness(fg.astype(np.uint8) * 255)
    _area_tmp, _per_tmp, _circ_tmp, _axis_tmp = _contour_metrics(fg.astype(np.uint8) * 255)
    if not force_circular and (_br < 0.55 or _ext < 0.30 or _circ_tmp < 0.45 or _axis_tmp < 0.45):
        # ← ZHANG: komponen memanjang → Zhang-Suen murni langsung dari fg
        return zhang_suen_thinning(fg).astype(np.uint8)

    # ============================================================
    # KHUSUS circular blob:
    # targetnya seperti gambar 2 -> loop harus mengikuti KONTUR LUAR,
    # bukan kontur inner hasil distance transform.
    # ============================================================
    if force_circular:
        # Guard tambahan:
        # meskipun cabang memaksa circular, stroke tebal memanjang yang OPEN
        # tetap harus diperlakukan sebagai skeleton normal, bukan contour luar.
        fg_u8 = (fg.astype(np.uint8) * 255)
        if _is_elongated_open_stroke(fg_u8):
            return zhang_suen_thinning(fg).astype(np.uint8)

        # PENTING:
        # fg yang masuk ke sini SUDAH hasil erosi 1px dari blok circular_blob.
        # Jadi permintaan dosen soal erosi tetap terpenuhi.
        #
        # Masalah hasil lama: kontur diambil dari INNER DT region,
        # sehingga loop jatuh terlalu ke dalam (seperti gambar 1).
        # Solusi: setelah erosi, ambil OUTER contour langsung dari fg hasil erosi.
        contours, _ = cv.findContours(fg_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return np.zeros_like(fg, dtype=np.uint8)

        skel_contour = np.zeros_like(fg, dtype=np.uint8)
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(skel_contour, [largest_contour], -1, 1, 1)

        # Jangan thinning lagi untuk circular blob, karena contour 1px ini
        # sudah cukup dan thinning/pruning justru bisa menggeser loop.
        return (skel_contour > 0).astype(np.uint8)

    dist = distance_transform_edt(fg)
    dist_vals = dist[fg]
    if len(dist_vals) == 0:
        return np.zeros_like(fg, dtype=np.uint8)

    thresh_val = np.percentile(dist_vals, dt_percentile)
    thresh_val = max(thresh_val, min_radius)

    inner = (dist >= thresh_val) & fg
    inner_u8 = (inner.astype(np.uint8) * 255)

    kernel_ero = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    inner_eroded = cv.erode(inner_u8, kernel_ero, iterations=1)
    src_for_cnt = inner_eroded if np.any(inner_eroded > 0) else inner_u8

    contours, _ = cv.findContours(src_for_cnt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    skel_contour = np.zeros_like(fg, dtype=np.uint8)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(skel_contour, [largest_contour], -1, 1, 1)

    if skel_contour.sum() < 5:
        fg_u8 = (fg.astype(np.uint8) * 255)
        contours_fb, _ = cv.findContours(fg_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours_fb) > 0:
            largest_fb = max(contours_fb, key=cv.contourArea)
            cv.drawContours(skel_contour, [largest_fb], -1, 1, 1)

    skel_final = zhang_suen_thinning(skel_contour > 0).astype(np.uint8)

    if skel_final.sum() == 0:
        skel_final = zhang_suen_thinning(fg).astype(np.uint8)

    return skel_final

def _largest_filled_contour_mask(mask_bool):
    """Mask isi (filled) dari kontur terbesar pada komponen blob."""
    mu8 = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv.findContours(mu8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mu8, dtype=np.uint8)
    ring = np.zeros_like(mu8, dtype=np.uint8)
    if not cnts:
        return filled.astype(bool), ring.astype(bool)
    c = max(cnts, key=cv.contourArea)
    cv.drawContours(filled, [c], -1, 255, thickness=-1)
    cv.drawContours(ring, [c], -1, 255, thickness=1)
    return (filled > 0), (ring > 0)


def enforce_hollow_circular_blob(skel_u8, blob_mask_bool):
    """
    Paksa skeleton circular blob menjadi loop kosong:
    - area dalam blob diisi 0
    - hanya kontur / ring 1px yang dipertahankan
    """
    if np.count_nonzero(skel_u8) == 0 or np.count_nonzero(blob_mask_bool) == 0:
        return (skel_u8 > 0).astype(np.uint8)

    filled_mask, ring_mask = _largest_filled_contour_mask(blob_mask_bool)
    if not np.any(filled_mask):
        return (skel_u8 > 0).astype(np.uint8)

    interior_mask = filled_mask & (~ring_mask)
    out = (skel_u8 > 0).astype(np.uint8)
    out[interior_mask] = 0
    out[ring_mask] = 1
    return out.astype(np.uint8)



def remove_inner_lines_from_closed_loops(skel_u8, min_loop_len=8, min_loop_area=12):
    """
    Hapus garis skeleton yang berada DI DALAM loop tertutup,
    tetapi pertahankan garis loop luarnya.

    Cocok untuk kasus seperti area yang dilingkari merah:
    ada loop, tapi masih ada garis kecil/garis tengah di dalamnya.
    """
    sk = (skel_u8 > 0).astype(np.uint8)
    ys, xs = np.where(sk > 0)
    coords = list(zip(xs.tolist(), ys.tolist()))
    if len(coords) == 0:
        return sk.astype(np.uint8)

    coord_set = set(coords)

    # Bangun graph piksel skeleton
    G = nx.Graph()
    for p in coords:
        G.add_node(p)

    for (x, y) in coords:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb in coord_set:
                    G.add_edge((x, y), nb)

    cycles = nx.cycle_basis(G)
    if not cycles:
        return sk.astype(np.uint8)

    out = sk.copy()

    for cyc in cycles:
        # loop terlalu kecil diabaikan
        if len(cyc) < min_loop_len:
            continue

        pts = np.array(cyc, dtype=np.int32)

        # urutkan titik cycle mengelilingi centroid
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        pts = pts[np.argsort(ang)]

        contour = pts.reshape((-1, 1, 2))
        area = cv.contourArea(contour)

        # loop terlalu kecil / noise diabaikan
        if area < min_loop_area:
            continue

        filled = np.zeros_like(sk, dtype=np.uint8)
        cv.fillPoly(filled, [contour], 1)

        cycle_mask = np.zeros_like(sk, dtype=np.uint8)
        cycle_mask[pts[:, 1], pts[:, 0]] = 1

        # bagian dalam loop = area isi - garis cycle itu sendiri
        inside_mask = (filled > 0) & (~(cycle_mask > 0))

        # Hapus semua garis skeleton di dalam loop
        out[inside_mask] = 0

    return (out > 0).astype(np.uint8)


def get_local_circular_core_mask(fg_bool, min_radius=2.0, dt_percentile=60):
    """
    Ambil area inti lokal dari blob bulat (bukan seluruh connected component).
    Area ini hanya dipakai untuk menghapus garis di bagian dalam lingkaran.
    """
    fg = fg_bool.astype(bool)
    if fg.sum() == 0:
        return np.zeros_like(fg, dtype=bool)

    dist = distance_transform_edt(fg)
    vals = dist[fg]
    if len(vals) == 0:
        return np.zeros_like(fg, dtype=bool)

    thr = max(np.percentile(vals, dt_percentile), min_radius)
    inner = (dist >= thr) & fg
    if not np.any(inner):
        return np.zeros_like(fg, dtype=bool)

    # Ambil komponen inner yang memuat puncak DT tertinggi
    y0, x0 = np.unravel_index(np.argmax(dist), dist.shape)
    nlab, lab = cv.connectedComponents(inner.astype(np.uint8), connectivity=8)

    lab_id = int(lab[y0, x0])
    if lab_id == 0:
        areas = [(i, int((lab == i).sum())) for i in range(1, nlab)]
        if not areas:
            return np.zeros_like(fg, dtype=bool)
        lab_id = max(areas, key=lambda t: t[1])[0]

    return (lab == lab_id)


def remove_inner_line_only(skel_u8, local_core_mask_bool):
    """
    Hanya hapus garis skeleton di bagian dalam blob bulat.
    Kontur/skeleton luar yang sudah ada dibiarkan apa adanya.
    """
    out = (skel_u8 > 0).astype(np.uint8)
    if out.sum() == 0 or np.count_nonzero(local_core_mask_bool) == 0:
        return out

    core_u8 = (local_core_mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv.findContours(core_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not cnts:
        return out

    filled = np.zeros_like(core_u8, dtype=np.uint8)
    ring   = np.zeros_like(core_u8, dtype=np.uint8)

    c = max(cnts, key=cv.contourArea)
    cv.drawContours(filled, [c], -1, 255, thickness=-1)
    cv.drawContours(ring,   [c], -1, 255, thickness=1)

    interior = (filled > 0) & (~(ring > 0))
    out[interior] = 0
    return out.astype(np.uint8)



def keep_largest_skel_component(skel_u8):
    sk = (skel_u8 > 0).astype(np.uint8)
    if sk.sum() == 0:
        return sk
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(sk, connectivity=8)
    if n_labels <= 1:
        return (sk * 255).astype(np.uint8)
    areas = stats[1:, cv.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    out = np.zeros_like(sk, dtype=np.uint8)
    out[labels == largest_label] = 255
    return out


def prune_spurs_graph(skel_u8, max_spur_len=12):
    sk = (skel_u8 > 0).copy()
    if sk.sum() == 0:
        return (sk * 255).astype(np.uint8)

    ys, xs = np.where(sk)
    coords = list(zip(xs.tolist(), ys.tolist()))
    coord_set = set(coords)

    degree = {}
    neighbors_map = {}
    for (x, y) in coords:
        nbs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb in coord_set:
                    nbs.append(nb)
        degree[(x, y)] = len(nbs)
        neighbors_map[(x, y)] = nbs

    endpoints = [p for p in coords if degree[p] == 1]
    to_remove = set()

    for ep in endpoints:
        if ep in to_remove:
            continue
        branch = [ep]
        prev = None
        cur = ep
        while True:
            nbs = [n for n in neighbors_map[cur] if n != prev and n not in to_remove]
            if not nbs:
                break
            if degree[cur] >= 3 and cur != ep:
                break
            if len(nbs) == 1:
                nxt = nbs[0]
                branch.append(nxt)
                if degree[nxt] >= 3:
                    break
                if degree[nxt] == 1 and nxt != ep:
                    if len(branch) <= max_spur_len:
                        to_remove.update(branch)
                    break
                prev = cur
                cur = nxt
            else:
                break
        if len(branch) <= max_spur_len and branch[-1] != ep:
            if degree.get(branch[-1], 0) >= 3:
                to_remove.update(branch)

    out = sk.copy()
    for (x, y) in to_remove:
        out[y, x] = False
    return (out.astype(np.uint8) * 255)


def prune_spurs(skel_u8, max_spur_len=8):
    sk = (skel_u8 > 0).copy()
    for _ in range(max_spur_len + 2):
        if sk.sum() == 0:
            break
        from scipy.ndimage import convolve
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        deg = convolve(sk.astype(np.uint8), kernel, mode='constant', cval=0)
        deg = deg * sk.astype(np.uint8)
        endpoints = (deg == 1) & sk
        if not endpoints.any():
            break
        sk[endpoints] = False
    return (sk.astype(np.uint8) * 255)


def combine_component_skeletons(skel_list, shape_hw):
    out = np.zeros(shape_hw, dtype=np.uint8)
    for s in skel_list:
        out |= (s > 0).astype(np.uint8)
    return out


def evaluasi_skeleton_lengkap(skeleton, edge, dt_edge, component_types):
    panjang = np.sum(skeleton > 0)
    overlap = np.logical_and(skeleton > 0, edge > 0).sum()
    dist_vals = dt_edge[skeleton > 0]
    dt_mean = float(dist_vals.mean()) if dist_vals.size else 0.0
    dt_median = float(np.median(dist_vals)) if dist_vals.size else 0.0
    dt_min = float(dist_vals.min()) if dist_vals.size else 0.0
    dt_max = float(dist_vals.max()) if dist_vals.size else 0.0
    num_labels, labels = cv.connectedComponents(skeleton.astype(np.uint8))
    return {
        'panjang_skeleton': int(panjang),
        'jumlah_komponen_skeleton': int(num_labels - 1),
        'jumlah_komponen_input': int(len(component_types)),
        'overlap_dengan_edge': int(overlap),
        'dt_mean_skel_to_edge': dt_mean,
        'dt_median_skel_to_edge': dt_median,
        'dt_min_skel_to_edge': dt_min,
        'dt_max_skel_to_edge': dt_max,
        'tipe_diacritic': component_types.count('diacritic'),
        'tipe_circular_blob': component_types.count('circular_blob'),
        'tipe_normal': component_types.count('normal'),
    }


def detect_blob_type(binary_mat, cy, baseline_y, median_body_h, r_stroke=0.0):
    """
    Klasifikasi komponen: 'diacritic', 'circular_blob', atau 'normal'.

    Threshold menggunakan skala n * r_stroke^2 (soft-scalable):
      r_stroke = estimasi jari-jari rata-rata stroke (dari DT median seluruh gambar,
                 diteruskan dari luar); jika tidak tersedia, di-fallback dari median_body_h.

      Konstanta natural yang dipakai:
        N_DIAC  = π/4  ≈ 0.785  → area lingkaran jari-jari r (satu stroke tipis membulat)
        N_BLOB  = 3π   ≈ 9.42   → area lingkaran jari-jari 3r (blob yang lebih besar dari stroke)
        N_MAX   = 20π  ≈ 62.8   → batas atas blob (komponen sangat besar = huruf normal)

      Artinya:
        diacritic  : area  <  N_DIAC * r^2  (lebih kecil dari satu disk stroke)
        circular   : N_DIAC * r^2 ≤ area ≤ N_MAX * r^2  (dari satu disk s.d. ~4.5r radius)
        normal     : area  >  N_MAX * r^2

      Nilai N bisa disesuaikan sebagai "konstanta natural" tanpa mengubah logika.
    """
    # --- Konstanta natural (n) ---
    N_DIAC = math.pi / 4.0     # ≈ 0.785  — satu disk jari-jari r
    N_BLOB = 3.0 * math.pi     # ≈ 9.42   — disk jari-jari ~3r (blob nyata)
    N_MAX  = 20.0 * math.pi    # ≈ 62.83  — disk jari-jari ~4.5r (batas atas blob)

    area_px = cv.countNonZero(binary_mat)
    area, per, circ, axis = _contour_metrics(binary_mat)
    dy = abs(float(cy) - float(baseline_y))

    # --- Estimasi r_stroke jika tidak diberikan ---
    if r_stroke <= 0.0:
        # Fallback: estimasi dari DT pada foreground komponen ini
        _fg = (binary_mat > 0)
        if _fg.sum() > 0:
            _dist = distance_transform_edt(_fg)
            r_stroke = float(np.median(_dist[_fg]))
        if r_stroke <= 0.0:
            # Last fallback: proporsi median_body_h (stroke ≈ 1/8 tinggi huruf)
            r_stroke = max(2.0, median_body_h / 8.0)

    r2 = r_stroke * r_stroke  # r^2 — satuan dasar area

    # --- Threshold area scalable ---
    area_diac_max = N_DIAC * r2   # batas atas diakritik
    area_blob_max = N_MAX  * r2   # batas atas blob (di atas ini = huruf normal)

    # --- Threshold jarak vertikal scalable ---
    # diakritik harus jauh dari baseline: dy > k_dy * r_stroke
    # k_dy = 2.5 → jarak > 2.5 kali lebar stroke = cukup "melayang"
    k_dy   = 2.5
    dy_min = k_dy * r_stroke          # minimum dy untuk dianggap diakritik
    dy_max = max(dy_min, 0.35 * median_body_h)  # tetap hormat ke median_body_h

    # --- SOFT SCORE untuk circular_blob ---
    # Alih-alih hard cut, kita pakai skor kontinyu:
    # score_circ = sigmoid-like dari (circ - circ_center) / circ_width
    # Ini membuat batas "soft" — komponen yang hampir bundar tetap bisa lolos
    def _soft_gate(val, center, width):
        """Sigmoid lembut: 1 jika val jauh di atas center, 0 jika jauh di bawah."""
        return 1.0 / (1.0 + math.exp(-(val - center) / (width + 1e-9)))

    bbox_ratio, extent, bw_bbox, bh_bbox = _bbox_extent_roundness(binary_mat)

    # --- Klasifikasi ---

    # 1) DIAKRITIK: area kecil (< N_DIAC * r^2) DAN melayang jauh dari baseline
    if area_px < area_diac_max and dy > dy_max:
        return 'diacritic', circ, axis, dy

    # 2) CIRCULAR BLOB: area dalam rentang scalable + bentuk cukup bulat (soft)
    #    Gunakan soft gate sehingga tidak ada batas keras
    if area_px <= area_blob_max:
        # Skor kebulatan: rata-rata soft gate untuk circ, axis, bbox_ratio
        # center=0.40 (sedang), width=0.08 (transisi gradual)
        score_circ  = _soft_gate(circ,       center=0.40, width=0.08)
        score_axis  = _soft_gate(axis,       center=0.40, width=0.08)
        score_bbox  = _soft_gate(bbox_ratio, center=0.38, width=0.08)
        score_round = (score_circ + score_axis + score_bbox) / 3.0

        # Guard tambahan:
        # komponen yang tampak cukup bulat secara global kadang sebenarnya
        # hanyalah stroke tebal memanjang yang terbuka. Kasus seperti ini
        # harus tetap dianggap normal.
        stroke_veto = _is_elongated_open_stroke(binary_mat)

        # Syarat: skor rata-rata >= 0.5 (artinya mayoritas dimensi "cukup bulat")
        if score_round >= 0.50 and (not stroke_veto):
            # Pastikan tidak terlalu jauh dari baseline (dy scalable)
            dy_blob_max = dy_max * 2.0  # blob boleh sedikit lebih jauh dari diakritik
            if dy <= dy_blob_max:
                return 'circular_blob', circ, axis, dy

    # 3) NORMAL: semua yang tidak masuk dua kategori di atas
    return 'normal', circ, axis, dy


def detect_blob_type_param(binary_mat, cy, baseline_y, median_body_h, r_stroke=0.0,
                           mode="scaled_soft",
                           N_DIAC=math.pi/4.0,
                           N_MAX=20.0*math.pi,
                           fixed_diac_area=None,
                           fixed_blob_area=None):
    area_px = int(cv.countNonZero(binary_mat))
    _, _, circ, axis = _contour_metrics(binary_mat)
    bbox_ratio, extent, bw_bbox, bh_bbox = _bbox_extent_roundness(binary_mat)
    dy = abs(float(cy) - float(baseline_y))

    if r_stroke <= 0.0:
        _fg = (binary_mat > 0)
        if _fg.sum() > 0:
            _dist = distance_transform_edt(_fg)
            r_stroke = float(np.median(_dist[_fg]))
        if r_stroke <= 0.0:
            r_stroke = max(2.0, median_body_h / 8.0)

    r2 = float(r_stroke * r_stroke)
    dy_max = max(2.5 * float(r_stroke), 0.35 * float(median_body_h))

    def _soft_gate(val, center, width):
        return 1.0 / (1.0 + math.exp(-(val - center) / (width + 1e-9)))

    score_circ = _soft_gate(circ,       center=0.40, width=0.08)
    score_axis = _soft_gate(axis,       center=0.40, width=0.08)
    score_bbox = _soft_gate(bbox_ratio, center=0.38, width=0.08)
    score_round = (score_circ + score_axis + score_bbox) / 3.0

    if mode == "fixed_soft":
        area_diac_max = float(fixed_diac_area)
        area_blob_max = float(fixed_blob_area)
    else:
        area_diac_max = float(N_DIAC * r2)
        area_blob_max = float(N_MAX  * r2)

    aspect = float(max(bw_bbox, bh_bbox) / (min(bw_bbox, bh_bbox) + 1e-9))
    stroke_veto = (
        ((aspect >= 2.4) and (bbox_ratio <= 0.42)) or
        ((circ <= 0.32) and (extent <= 0.20) and (bbox_ratio <= 0.55))
    )

    if area_px < area_diac_max and dy > dy_max:
        label = "diacritic"
    else:
        round_ok = False
        if area_px <= area_blob_max:
            if mode in ("scaled_soft", "fixed_soft"):
                round_ok = (score_round >= 0.50)
            elif mode == "scaled_hard":
                round_ok = (circ >= 0.40 and axis >= 0.40 and bbox_ratio >= 0.38)
            elif mode == "scaled_area_only":
                round_ok = True

        if round_ok and (not stroke_veto) and dy <= (2.0 * dy_max):
            label = "circular_blob"
        else:
            label = "normal"

    return {
        "label": label,
        "area_px": area_px,
        "circ": float(circ),
        "axis": float(axis),
        "bbox_ratio": float(bbox_ratio),
        "extent": float(extent),
        "dy": float(dy),
        "dy_max": float(dy_max),
        "r_stroke": float(r_stroke),
        "r2": float(r2),
        "area_over_r2": float(area_px / (r2 + 1e-9)),
        "dy_over_r": float(dy / (r_stroke + 1e-9)),
        "score_round": float(score_round),
        "area_diac_max": float(area_diac_max),
        "area_blob_max": float(area_blob_max),
    }


def _collect_component_features(components, baseline_y, median_body_h, r_stroke_global):
    rows = []
    for idx, comp in enumerate(components):
        binary_mat = ((comp.mat == RASMVAL).astype(np.uint8) * 255)
        res = detect_blob_type_param(
            binary_mat, comp.centroid[1], baseline_y, median_body_h,
            r_stroke=r_stroke_global, mode="scaled_soft"
        )
        x, y, w, h = comp.rect
        rows.append({
            "idx": idx,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": int(comp.centroid[0]), "cy": int(comp.centroid[1]),
            **res
        })
    return rows




def _evaluate_config(feature_rows, config_name, mode, N_DIAC, N_MAX,
                     fixed_diac_area=None, fixed_blob_area=None):
    out_rows = []
    for f in feature_rows:
        area_diac_max = fixed_diac_area if mode == "fixed_soft" else N_DIAC * f["r2"]
        area_blob_max = fixed_blob_area if mode == "fixed_soft" else N_MAX * f["r2"]

        if f["area_px"] < area_diac_max and f["dy"] > f["dy_max"]:
            label = "diacritic"
        else:
            round_ok = False
            if f["area_px"] <= area_blob_max:
                if mode in ("scaled_soft", "fixed_soft"):
                    round_ok = (f["score_round"] >= 0.50)
                elif mode == "scaled_hard":
                    round_ok = (f["circ"] >= 0.40 and f["axis"] >= 0.40 and f["bbox_ratio"] >= 0.38)
                elif mode == "scaled_area_only":
                    round_ok = True

            if round_ok and f["dy"] <= 2.0 * f["dy_max"]:
                label = "circular_blob"
            else:
                label = "normal"

        row = dict(f)
        row["config_name"] = config_name
        row["label"] = label
        row["cfg_area_diac_max"] = float(area_diac_max)
        row["cfg_area_blob_max"] = float(area_blob_max)
        out_rows.append(row)
    return out_rows


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _plot_threshold_curve(imagename, r_stroke_global):
    r_vals = np.linspace(1.0, max(8.0, r_stroke_global * 4.0), 200)
    fixed_diac = (math.pi / 4.0) * (r_stroke_global ** 2)
    fixed_blob = (20.0 * math.pi) * (r_stroke_global ** 2)

    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, (math.pi/4.0) * (r_vals ** 2), label=r"Scaled diacritic: $\pi/4 \cdot r^2$")
    plt.plot(r_vals, (20.0*math.pi) * (r_vals ** 2), label=r"Scaled blob max: $20\pi \cdot r^2$")
    plt.axhline(fixed_diac, linestyle="--", label=f"Fixed diacritic = {fixed_diac:.1f} px")
    plt.axhline(fixed_blob, linestyle="--", label=f"Fixed blob max = {fixed_blob:.1f} px")
    plt.axvline(r_stroke_global, linestyle=":", label=f"r_stroke saat ini = {r_stroke_global:.2f} px")
    plt.xlabel("r_stroke")
    plt.ylabel("Threshold area (px)")
    plt.title("Perbandingan threshold tetap vs threshold scalable")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{imagename}_fig_threshold_curve.png", dpi=200)
    plt.close()


def _plot_normalized_scatter(imagename, default_rows, r_stroke_global, median_body_h):
    dy_max_global = max(2.5 * r_stroke_global, 0.35 * median_body_h)
    y_thr = dy_max_global / (r_stroke_global + 1e-9)

    classes = ["diacritic", "circular_blob", "normal"]
    colors = {"diacritic": "tab:red", "circular_blob": "tab:green", "normal": "tab:blue"}

    plt.figure(figsize=(8, 6))
    for cls in classes:
        xs = [r["area_over_r2"] for r in default_rows if r["label"] == cls]
        ys = [r["dy_over_r"] for r in default_rows if r["label"] == cls]
        if xs:
            plt.scatter(xs, ys, s=22, alpha=0.75, label=cls, c=colors[cls])

    plt.axvline(math.pi/4.0, linestyle="--", label=r"$N_{DIAC}=\pi/4$")
    plt.axvline(20.0*math.pi, linestyle="--", label=r"$N_{MAX}=20\pi$")
    plt.axhline(y_thr, linestyle=":", label=r"$dy_{max}/r$")
    plt.xlabel(r"Area ternormalisasi: $A / r^2$")
    plt.ylabel(r"Jarak ternormalisasi: $dy / r$")
    plt.title("Sebaran komponen pada ruang keputusan yang dinormalisasi")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{imagename}_fig_normalized_scatter.png", dpi=200)
    plt.close()


def _plot_sensitivity(imagename, feature_rows):
    ndiac_values = [math.pi/8.0, math.pi/4.0, math.pi/2.0, math.pi]
    nmax_values  = [10.0*math.pi, 20.0*math.pi, 30.0*math.pi]

    ndiac_counts = {"diacritic": [], "circular_blob": [], "normal": []}
    for nd in ndiac_values:
        rows = _evaluate_config(feature_rows, f"N_DIAC={nd:.4f}", "scaled_soft", nd, 20.0*math.pi)
        for cls in ndiac_counts:
            ndiac_counts[cls].append(sum(1 for r in rows if r["label"] == cls))

    nmax_counts = {"diacritic": [], "circular_blob": [], "normal": []}
    for nm in nmax_values:
        rows = _evaluate_config(feature_rows, f"N_MAX={nm:.4f}", "scaled_soft", math.pi/4.0, nm)
        for cls in nmax_counts:
            nmax_counts[cls].append(sum(1 for r in rows if r["label"] == cls))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    x1 = np.arange(len(ndiac_values))
    for cls in ["diacritic", "circular_blob", "normal"]:
        plt.plot(x1, ndiac_counts[cls], marker="o", label=cls)
    plt.xticks(x1, ["π/8", "π/4", "π/2", "π"])
    plt.xlabel("Variasi N_DIAC")
    plt.ylabel("Jumlah komponen")
    plt.title("Sensitivitas terhadap N_DIAC")
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    x2 = np.arange(len(nmax_values))
    for cls in ["diacritic", "circular_blob", "normal"]:
        plt.plot(x2, nmax_counts[cls], marker="o", label=cls)
    plt.xticks(x2, ["10π", "20π", "30π"])
    plt.xlabel("Variasi N_MAX")
    plt.ylabel("Jumlah komponen")
    plt.title("Sensitivitas terhadap N_MAX")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{imagename}_fig_sensitivity.png", dpi=200)
    plt.close()


def _draw_label_overlay(gray_image, components, labels, out_path, title_text):
    if len(gray_image.shape) == 2:
        canvas = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
    else:
        canvas = gray_image.copy()

    color_map = {
        "diacritic": (0, 0, 255),
        "circular_blob": (0, 180, 0),
        "normal": (255, 0, 0),
    }

    for comp, lab in zip(components, labels):
        x, y, w, h = comp.rect
        cx, cy = comp.centroid
        color = color_map.get(lab, (255, 255, 255))
        cv.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        tag = f"{lab[0].upper()}"
        cv.putText(canvas, tag, (max(0, x), max(12, y - 3)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv.LINE_AA)
        cv.circle(canvas, (int(cx), int(cy)), 1, color, -1)

    plt.figure(figsize=(10, 4))
    plt.imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    plt.title(title_text)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_ablation_formula_vs_constants(components, baseline_y, median_body_h,
                                      r_stroke_global, gray_image, imagename):
    feature_rows = _collect_component_features(
        components, baseline_y, median_body_h, r_stroke_global
    )

    fixed_diac_area = (math.pi / 4.0) * (r_stroke_global ** 2)
    fixed_blob_area = (20.0 * math.pi) * (r_stroke_global ** 2)

    configs = [
        ("scaled_soft_default", "scaled_soft", math.pi/4.0, 20.0*math.pi, None, None),
        ("fixed_soft_reference", "fixed_soft", math.pi/4.0, 20.0*math.pi, fixed_diac_area, fixed_blob_area),
        ("scaled_hard_roundness", "scaled_hard", math.pi/4.0, 20.0*math.pi, None, None),
        ("scaled_area_only", "scaled_area_only", math.pi/4.0, 20.0*math.pi, None, None),
        ("ndiac_pi_over_8", "scaled_soft", math.pi/8.0, 20.0*math.pi, None, None),
        ("ndiac_pi_over_2", "scaled_soft", math.pi/2.0, 20.0*math.pi, None, None),
        ("nmax_10pi", "scaled_soft", math.pi/4.0, 10.0*math.pi, None, None),
        ("nmax_30pi", "scaled_soft", math.pi/4.0, 30.0*math.pi, None, None),
    ]

    summary_rows = []

    default_rows = _evaluate_config(
        feature_rows, "scaled_soft_default", "scaled_soft", math.pi/4.0, 20.0*math.pi
    )
    default_labels = [r["label"] for r in default_rows]

    labels_by_cfg = {}
    for cfg_name, mode, nd, nm, fda, fba in configs:
        rows = _evaluate_config(feature_rows, cfg_name, mode, nd, nm, fda, fba)
        labels = [r["label"] for r in rows]
        labels_by_cfg[cfg_name] = labels

        agreement = sum(int(a == b) for a, b in zip(labels, default_labels)) / max(1, len(labels))

        summary_rows.append({
            "config_name": cfg_name,
            "mode": mode,
            "N_DIAC": float(nd),
            "N_MAX": float(nm),
            "fixed_diac_area": "" if fda is None else float(fda),
            "fixed_blob_area": "" if fba is None else float(fba),
            "count_diacritic": int(sum(l == "diacritic" for l in labels)),
            "count_circular_blob": int(sum(l == "circular_blob" for l in labels)),
            "count_normal": int(sum(l == "normal" for l in labels)),
            "agreement_vs_default": float(agreement),
        })

    component_table = []
    for i, feat in enumerate(feature_rows):
        row = {
            "idx": feat["idx"],
            "x": feat["x"], "y": feat["y"], "w": feat["w"], "h": feat["h"],
            "cx": feat["cx"], "cy": feat["cy"],
            "area_px": feat["area_px"],
            "area_over_r2": feat["area_over_r2"],
            "dy": feat["dy"],
            "dy_over_r": feat["dy_over_r"],
            "circ": feat["circ"],
            "axis": feat["axis"],
            "bbox_ratio": feat["bbox_ratio"],
            "score_round": feat["score_round"],
        }
        for cfg_name in labels_by_cfg:
            row[f"label__{cfg_name}"] = labels_by_cfg[cfg_name][i]
        component_table.append(row)

    _write_csv(
        f"{imagename}_ablation_summary.csv",
        summary_rows,
        ["config_name", "mode", "N_DIAC", "N_MAX", "fixed_diac_area",
         "fixed_blob_area", "count_diacritic", "count_circular_blob",
         "count_normal", "agreement_vs_default"]
    )

    if component_table:
        _write_csv(
            f"{imagename}_component_labels.csv",
            component_table,
            list(component_table[0].keys())
        )

    _plot_threshold_curve(imagename, r_stroke_global)
    _plot_normalized_scatter(imagename, default_rows, r_stroke_global, median_body_h)
    _plot_sensitivity(imagename, feature_rows)

    _draw_label_overlay(
        gray_image, components, labels_by_cfg["scaled_soft_default"],
        f"{imagename}_overlay_scaled_soft_default.png",
        "Klasifikasi komponen - scaled soft default"
    )
    _draw_label_overlay(
        gray_image, components, labels_by_cfg["fixed_soft_reference"],
        f"{imagename}_overlay_fixed_soft_reference.png",
        "Klasifikasi komponen - fixed threshold reference"
    )

    print("\n[ABLATION] File keluaran:")
    print(f"  - {imagename}_ablation_summary.csv")
    print(f"  - {imagename}_component_labels.csv")
    print(f"  - {imagename}_fig_threshold_curve.png")
    print(f"  - {imagename}_fig_normalized_scatter.png")
    print(f"  - {imagename}_fig_sensitivity.png")
    print(f"  - {imagename}_overlay_scaled_soft_default.png")
    print(f"  - {imagename}_overlay_fixed_soft_reference.png")



def preserve_holes(fg255):
    bg = cv.bitwise_not(fg255)
    h, w = bg.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    bg_ff = bg.copy()
    cv.floodFill(bg_ff, mask, (0, 0), 0)
    holes = bg_ff
    return cv.bitwise_and(fg255, cv.bitwise_not(holes))



def thin_stroke_adaptive_boost(img_gray, base_bin, otsu_T, debug=False):
    """
    Rescue khusus stroke tipis.
    Ide:
    1) adaptive threshold multi-skala (coarse + fine)
    2) kandidat stroke tipis dari ridge terang + gradient lokal
    3) hanya dipakai kalau dekat foreground existing supaya noise liar tidak ikut
    """
    H, W = img_gray.shape

    # adaptive multi-skala: window kecil lebih sensitif ke stroke tipis
    th_fine_1 = cv.adaptiveThreshold(
        img_gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        15,
        -2
    )
    th_fine_2 = cv.adaptiveThreshold(
        img_gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        21,
        -4
    )

    # ridge terang / detail tipis
    blur = cv.GaussianBlur(img_gray, (0, 0), 1.0)
    lap = cv.Laplacian(blur, cv.CV_32F, ksize=3)
    grad = cv.morphologyEx(img_gray, cv.MORPH_GRADIENT,
                           cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

    # local mean & std untuk adaptasi "tipis tapi cukup terang"
    mean = cv.GaussianBlur(img_gray.astype('float32'), (0, 0), 3.0)
    mean2 = cv.GaussianBlur((img_gray.astype('float32') ** 2), (0, 0), 3.0)
    std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))

    # kandidat piksel tipis:
    # - lolos salah satu adaptive halus
    # - relatif terang terhadap mean lokal
    # - punya gradient atau respon ridge yang cukup
    cand_fine = ((th_fine_1 > 0) | (th_fine_2 > 0))
    bright_local = img_gray.astype('float32') >= (mean - 0.35 * std - 2.0)
    grad_thr = max(4.0, float(np.percentile(grad, 55)))
    ridge_thr = float(np.percentile(np.abs(lap), 60))
    detail_local = (grad.astype('float32') >= grad_thr) | (np.abs(lap) >= ridge_thr)

    # batasi agar tidak terlalu jauh dari threshold global
    near_otsu = img_gray >= max(0, int(otsu_T) - 22)

    thin_candidates = cand_fine & bright_local & detail_local & near_otsu

    # hanya pertahankan kandidat yang dekat dengan foreground existing
    base01 = (base_bin > 0).astype(np.uint8)
    k_near = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    near_base = cv.dilate(base01, k_near, iterations=1) > 0

    # juga izinkan dalam band teks horizontal
    y0, y1 = estimate_text_band((base01 * 255), row_frac=0.002, pad_y=60)
    text_band = np.zeros_like(base01, dtype=bool)
    text_band[y0:y1+1, :] = True

    thin_keep = thin_candidates & near_base & text_band

    # sambung sedikit biar stroke 1px tidak putus
    thin_keep = cv.morphologyEx(
        (thin_keep.astype(np.uint8) * 255),
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)),
        iterations=1
    ) > 0

    boosted = ((base01 > 0) | thin_keep).astype(np.uint8) * 255

    if debug:
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(th_fine_1, cmap='gray')
        plt.title("Fine adaptive 15")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(th_fine_2, cmap='gray')
        plt.title("Fine adaptive 21")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow((thin_keep.astype(np.uint8) * 255), cmap='gray')
        plt.title("Thin-stroke boost")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(boosted, cmap='gray')
        plt.title("Binary + boost")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return boosted


def safe_threshold_text(img_gray, otsu_bias_up=2, debug=False, return_gate=False):
    """
    Threshold yang lebih aman untuk teks tipis:
    - adaptive Gaussian dengan C lebih lembut
    - gate Otsu yang lebih longgar
    - UNION (OR) bukan AND supaya stroke yang lolos salah satu jalur tetap hidup
    - close ringan, TANPA open global
    - hole preservation tetap dijaga pada jalur final aman

    return_gate=True -> kembalikan juga hasil panel "Gate Otsu longgar"
                        agar bisa dipakai langsung sebagai binary final.
    """
    T, _ = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # hole map dari threshold yang sedikit dinaikkan, tapi tidak agresif
    Th_hole = min(255, int(T) + 18)
    _, th_hole = cv.threshold(img_gray, Th_hole, 255, cv.THRESH_BINARY)
    holes0 = get_holes_mask(th_hole)

    # adaptive dibuat lebih permisif agar stroke tipis tetap lolos
    th_adapt = cv.adaptiveThreshold(
        img_gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        31,     # lebih lokal, tidak terlalu meratakan area lebar
        -7      # lebih ringan dari -13
    )

    # gate Otsu dibuat longgar
    _, th_gate = cv.threshold(
        img_gray, min(255, int(T) + otsu_bias_up), 255, cv.THRESH_BINARY
    )

    # Simpan versi gate apa adanya, karena target mode ini memang persis panel gate
    th_gate_direct = th_gate.copy()

    # KUNCI: pakai OR, bukan AND
    th_main = cv.bitwise_or(th_adapt, th_gate)

    # close ringan untuk menyambung gap kecil; hindari open global karena bisa makan stroke
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    th_main = cv.morphologyEx(th_main, cv.MORPH_CLOSE, k_close, iterations=1)

    # BOOST khusus stroke tipis
    th_main = thin_stroke_adaptive_boost(img_gray, th_main, T, debug=debug)

    # jaga hole tetap hole pada jalur final aman
    th_main[holes0 == 255] = 0

    if debug:
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title("Gray input")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(th_adapt, cmap='gray')
        plt.title("Adaptive aman")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(th_gate_direct, cmap='gray')
        plt.title("Gate Otsu longgar")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(th_main, cmap='gray')
        plt.title("Final binary aman + thin boost")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    if return_gate:
        return th_main, T, th_gate_direct
    return th_main, T

def get_holes_mask(fg255):
    fg255 = ((fg255 > 0).astype(np.uint8) * 255)
    pad = cv.copyMakeBorder(fg255, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    inv = cv.bitwise_not(pad)
    h, w = inv.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = inv.copy()
    cv.floodFill(flood, mask, (0, 0), 0)
    holes = flood[1:-1, 1:-1]
    return holes



def estimate_text_band(bin255, row_frac=0.003, pad_y=55):
    bw = (bin255 > 0).astype(np.uint8)
    H, W = bw.shape
    row_sum = bw.sum(axis=1)
    thr = int(max(1, row_frac * W))
    ys = np.where(row_sum > thr)[0]
    if len(ys) == 0:
        return 0, H - 1
    y0 = max(0, int(ys.min()) - pad_y)
    y1 = min(H - 1, int(ys.max()) + pad_y)
    return y0, y1



def remove_speckle_noise_preserve_thin_strokes(bin255, gray_img, min_area=6,
                                               near_radius=9, band_pad=60,
                                               solidity_thr=0.22, debug=False,
                                               body_area=220, body_h=18,
                                               small_keep_area=18,
                                               keep_dot_area=12):
    """
    Hapus noise bintik kecil TANPA membuang stroke tipis.

    Perbaikan utama dibanding versi sebelumnya:
    - support dihitung dari KOMPONEN LAIN (self tidak ikut), sehingga
      speckle soliter tidak lagi dianggap punya tetangga.
    - ada body-mask agar titik/diakritik yang dekat badan huruf tetap aman.
    - aturan buang lebih tegas untuk blob kecil yang padat, soliter, dan jauh
      dari badan huruf.
    """
    bw = (bin255 > 0).astype(np.uint8)
    H, W = bw.shape
    y0, y1 = estimate_text_band(bin255, row_frac=0.002, pad_y=band_pad)

    band = np.zeros_like(bw, dtype=np.uint8)
    band[y0:y1+1, :] = 1

    n, labels, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)
    out = bw.copy()

    body_mask = np.zeros_like(bw, dtype=np.uint8)
    body_w_thr = max(24, int(0.035 * W))
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= body_area or h >= body_h or w >= body_w_thr:
            body_mask[labels == i] = 1

    k_near = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (2 * near_radius + 1, 2 * near_radius + 1)
    )
    near_body = cv.dilate(body_mask, k_near, iterations=1)
    p55 = float(np.percentile(gray_img, 55))
    border = 4

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > 80:
            continue

        comp = (labels == i).astype(np.uint8)
        if np.count_nonzero(comp & band) == 0:
            out[labels == i] = 0
            continue

        dil_comp = cv.dilate(comp, np.ones((3, 3), np.uint8), iterations=1)

        # SELF-EXCLUDED support: hanya hitung dukungan dari komponen lain.
        others = bw.copy()
        others[labels == i] = 0
        near_others = cv.dilate(others, k_near, iterations=1)
        support = np.count_nonzero((near_others > 0) & (dil_comp > 0))

        ys, xs = np.where(comp > 0)
        if len(xs) == 0:
            continue

        aspect = max(w, h) / max(1.0, min(w, h))
        fill_ratio = area / max(1.0, float(w * h))
        local_vals = gray_img[comp > 0]
        local_mean = float(local_vals.mean()) if local_vals.size else 0.0
        near_body_flag = np.count_nonzero((near_body > 0) & (comp > 0)) > 0
        border_touch = (
            x <= border or y <= border or
            (x + w) >= (W - border) or (y + h) >= (H - border)
        )

        keep = False
        if area >= small_keep_area:
            keep = True
        if near_body_flag and area >= keep_dot_area:
            keep = True
        if aspect >= 3.0 and max(w, h) >= 6:
            keep = True
        if support > 0:
            keep = True
        if fill_ratio < solidity_thr and area >= min_area:
            keep = True
        if (local_mean >= p55) and near_body_flag and (area >= min_area or aspect >= 2.4):
            keep = True

        # Blob kecil, padat, soliter, dan jauh dari badan huruf = noise.
        if area <= 12 and fill_ratio >= 0.32 and support == 0 and not near_body_flag:
            keep = False
        if area <= 8 and not near_body_flag and aspect < 2.4:
            keep = False
        if border_touch and area < 40 and support == 0 and not near_body_flag:
            keep = False

        if not keep:
            out[labels == i] = 0

    out = (out * 255).astype(np.uint8)

    if debug:
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(bin255, cmap='gray')
        plt.title("Before despeckle")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        band_vis = np.zeros((H, W), dtype=np.uint8)
        band_vis[y0:y1+1, :] = 255
        plt.imshow(band_vis, cmap='gray')
        plt.title("Text band")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(body_mask * 255, cmap='gray')
        plt.title("Body mask")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(out, cmap='gray')
        plt.title("After despeckle")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return out


def remove_residual_speckles_by_body_proximity(bin255, body_area=220, body_h=18,
                                               dist_keep=12, small_area_max=18,
                                               compact_fill_min=0.32, border=4,
                                               row_frac=0.002, pad_y=55,
                                               debug=False):
    """
    Cleanup tahap-2 untuk sisa noise yang masih lolos.
    Hanya menarget komponen kecil yang compact, jauh dari badan huruf,
    dan tidak terlihat seperti stroke tipis.
    """
    bw = (bin255 > 0).astype(np.uint8)
    H, W = bw.shape
    y0, y1 = estimate_text_band(bin255, row_frac=row_frac, pad_y=pad_y)

    n, labels, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)

    body_mask = np.zeros_like(bw, dtype=np.uint8)
    body_w_thr = max(24, int(0.035 * W))
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= body_area or h >= body_h or w >= body_w_thr:
            body_mask[labels == i] = 1

    near_body = cv.dilate(
        body_mask,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dist_keep + 1, 2 * dist_keep + 1)),
        iterations=1
    )

    out = bw.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > small_area_max:
            continue

        comp = (labels == i)
        fill = area / max(1.0, float(w * h))
        aspect = max(w, h) / max(1.0, min(w, h))
        inside_band = (y + h >= y0) and (y <= y1)
        near = np.count_nonzero(near_body[comp]) > 0
        border_touch = (
            x <= border or y <= border or
            (x + w) >= (W - border) or (y + h) >= (H - border)
        )

        remove = False
        if not inside_band:
            remove = True
        elif border_touch and not near:
            remove = True
        elif (not near) and area <= 8:
            remove = True
        elif (not near) and fill >= compact_fill_min and aspect < 2.6:
            remove = True

        if remove:
            out[comp] = 0

    out = (out * 255).astype(np.uint8)

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(bin255, cmap='gray')
        plt.title("Residual speckle - before")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(body_mask * 255, cmap='gray')
        plt.title("Body mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(out, cmap='gray')
        plt.title("Residual speckle - after")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return out


def extract_body_mask_binary(bin255, body_area=220, body_h=18, body_w_frac=0.035,
                             row_frac=0.002, pad_y=55, close_iter=0,
                             debug=False):
    """
    Mode body-only:
    pertahankan hanya connected component yang cukup besar / tinggi / lebar,
    sehingga output binary mengikuti BODY MASK yang bersih.

    Cocok ketika target pengguna adalah binary final yang bersih total seperti
    panel "Body mask", bukan binary yang masih menyimpan titik/diakritik kecil.
    """
    bw = (bin255 > 0).astype(np.uint8)
    if bw.sum() == 0:
        return (bw * 255).astype(np.uint8)

    H, W = bw.shape
    y0, y1 = estimate_text_band(bin255, row_frac=row_frac, pad_y=pad_y)

    n, labels, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)
    body_mask = np.zeros_like(bw, dtype=np.uint8)
    body_w_thr = max(24, int(body_w_frac * W))

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        inside_band = (y + h >= y0) and (y <= y1)
        if not inside_band:
            continue
        if area >= body_area or h >= body_h or w >= body_w_thr:
            body_mask[labels == i] = 1

    if close_iter > 0:
        body_mask = cv.morphologyEx(
            (body_mask * 255).astype(np.uint8),
            cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
            iterations=close_iter
        )
        body_mask = (body_mask > 0).astype(np.uint8)

    out = (body_mask * 255).astype(np.uint8)

    if debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(bin255, cmap='gray')
        plt.title("Binary sebelum body-only")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(body_mask * 255, cmap='gray')
        plt.title("Body mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(out, cmap='gray')
        plt.title("Binary final (body-only)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return out


def rescue_text_components(binary_raw, binary_clean, border=4, min_area=4, dilate_keep=5):
    """
    Pulihkan komponen dari binary awal jika:
    - komponen berada di pita teks
    - komponen dekat dengan hasil clean yang tersisa
    - komponen bukan noise border yang jelas
    """
    raw = (binary_raw > 0).astype(np.uint8)
    clean = (binary_clean > 0).astype(np.uint8)
    H, W = raw.shape

    y0, y1 = estimate_text_band(binary_raw, row_frac=0.003, pad_y=55)
    keep_zone = np.zeros_like(raw, dtype=np.uint8)
    keep_zone[y0:y1+1, :] = 1

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilate_keep + 1, 2 * dilate_keep + 1))
    near_clean = cv.dilate(clean, k, iterations=1)

    out = clean.copy()
    n, labels, stats, _ = cv.connectedComponentsWithStats(raw, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        if np.count_nonzero(comp & keep_zone) == 0:
            continue
        if x <= border or y <= border or (x + w) >= (W - border) or (y + h) >= (H - border):
            # tetap izinkan jika komponen ini jelas menyentuh teks yang tersisa
            if np.count_nonzero(comp & near_clean) == 0:
                continue
        if np.count_nonzero(comp & near_clean) > 0:
            out |= comp

    return (out * 255).astype(np.uint8)

def clean_noise_keep_text_band(bin255, row_frac=0.01, pad_y=35, border=12, min_area_inside=6):
    bin01 = (bin255 > 0).astype(np.uint8)
    H, W = bin01.shape
    row_sum = bin01.sum(axis=1)
    thr = int(row_frac * W)
    ys = np.where(row_sum > thr)[0]
    if len(ys) > 0:
        y0 = max(0, int(ys.min()) - pad_y)
        y1 = min(H - 1, int(ys.max()) + pad_y)
    else:
        y0, y1 = 0, H - 1
    out = (bin01 * 255).copy()
    out[:y0, :] = 0
    out[y1+1:, :] = 0
    out01 = (out > 0).astype(np.uint8)
    n, labels, stats, cents = cv.connectedComponentsWithStats(out01, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x <= border or y <= border or (x + w) >= (W - border) or (y + h) >= (H - border):
            out[labels == i] = 0
            continue
        if area < min_area_inside:
            out[labels == i] = 0
    return out


def remove_noise_by_proximity(bin255, body_area=1200, body_h=26,
                              dot_area_max=900, dist_keep=20, border=12):
    bw = (bin255 > 0).astype(np.uint8)
    H, W = bw.shape
    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)
    body = np.zeros_like(bw)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= body_area or h >= body_h:
            body[labels == i] = 1
    if body.sum() == 0:
        return bin255
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*dist_keep+1, 2*dist_keep+1))
    near_body = cv.dilate(body, k, iterations=1)
    out = (bw * 255).copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x <= border or y <= border or (x + w) >= (W - border) or (y + h) >= (H - border):
            out[labels == i] = 0
            continue
        if area <= dot_area_max:
            if np.count_nonzero(near_body[labels == i]) == 0:
                out[labels == i] = 0
    return out


def separate_connected_dots_aggressive(bin255, max_component_area=400, erosion_size=2):
    bw = (bin255 > 0).astype(np.uint8)
    H, W = bw.shape
    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)
    out = bin255.copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area <= max_component_area:
            component_mask = (labels == i).astype(np.uint8) * 255
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (erosion_size, erosion_size))
            eroded = cv.erode(component_mask, kernel, iterations=1)
            out[labels == i] = 0
            out = cv.bitwise_or(out, eroded)
    return out


def remove_small_isolated_components(bin255, min_area=15, min_width=3, min_height=3):
    bw = (bin255 > 0).astype(np.uint8)
    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)
    out = (bw * 255).copy()
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < min_width or h < min_height:
            out[labels == i] = 0
    return out


def split_left_diacritic_pairs(gray255, left_frac=0.35, area_max=1200):
    bw = (gray255 > 0).astype(np.uint8)
    H, W = bw.shape
    n, labels, stats, _ = cv.connectedComponentsWithStats(bw, connectivity=8)
    out = (bw * 255).copy()
    candidates = [
        (cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)), 1),
        (cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), 1),
        (cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)), 2),
    ]
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > area_max:
            continue
        if x > int(left_frac * W):
            continue
        comp = (labels == i).astype(np.uint8) * 255
        best = None
        for ker, it in candidates:
            er = cv.erode(comp, ker, iterations=it)
            n2, lab2, st2, _ = cv.connectedComponentsWithStats((er > 0).astype(np.uint8), connectivity=8)
            if n2 - 1 >= 2:
                best = (ker, it, er, lab2, st2, n2)
                break
        if best is None:
            continue
        ker, it, er, lab2, st2, n2 = best
        out[labels == i] = 0
        recon = np.zeros_like(comp)
        for j in range(1, n2):
            sub = (lab2 == j).astype(np.uint8) * 255
            dil = cv.dilate(sub, ker, iterations=it)
            dil = cv.bitwise_and(dil, comp)
            recon = cv.bitwise_or(recon, dil)
        out = cv.bitwise_or(out, recon)
    return out


filename= sys.argv[1]
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
resz = cv.resize(image, (RESIZE_FACTOR*image.shape[1], RESIZE_FACTOR*image.shape[0]), interpolation=cv.INTER_LINEAR)
image= resz.copy()
image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
img = clahe.apply(image_gray)

# Threshold yang lebih aman agar stroke tipis tidak hilang
gray_main, T, gray_gate = safe_threshold_text(
    img, otsu_bias_up=2, debug=True, return_gate=True
)
Th = min(255, int(T) + 18)

if USE_GATE_OTSU_BINARY:
    print("Mode Gate Otsu aktif: binary final langsung mengikuti panel 'Gate Otsu longgar' agar diakritik tetap dipertahankan...")
    gray = gray_gate.copy()

    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.imshow(gray_main, cmap='gray')
    plt.title("Final binary aman + thin boost")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(gray, cmap='gray')
    plt.title("Binary final = Gate Otsu longgar")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    gray = gray_main.copy()
    gray_raw_preserve = gray.copy()

    # gray = split_left_diacritic_pairs(gray, left_frac=0.35, area_max=1200)  # dimatikan: terlalu agresif
    gray = clean_noise_keep_text_band(gray, row_frac=0.002, pad_y=55, border=4, min_area_inside=3)
    # gray = separate_connected_dots_aggressive(gray, max_component_area=160, erosion_size=2)  # dimatikan: bisa memutus huruf

    kernel_dot = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    bw_temp = (gray > 0).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(bw_temp, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 60 and h > w * 2.2:
            component = (labels == i).astype(np.uint8) * 255
            eroded = cv.erode(component, kernel_dot, iterations=1)
            gray[labels == i] = 0
            gray = cv.bitwise_or(gray, eroded)

    print("Menghapus noise kecil (super aman)...")
    gray = remove_small_isolated_components(gray, min_area=3, min_width=1, min_height=1)

    if USE_BODY_MASK_BINARY:
        print("Mode body-only aktif: binary final dibuat langsung dari body mask...")
        gray = extract_body_mask_binary(
            gray,
            body_area=BODY_MASK_AREA,
            body_h=BODY_MASK_H,
            body_w_frac=BODY_MASK_W_FRAC,
            row_frac=BODY_MASK_ROW_FRAC,
            pad_y=BODY_MASK_PAD_Y,
            close_iter=BODY_MASK_CLOSE_ITER,
            debug=True
        )
    else:
        print("Menghapus dot yang tidak terhubung (super aman)...")
        # gray = remove_noise_by_proximity(gray, body_area=1400, body_h=28, dot_area_max=220, dist_keep=12, border=8)  # dimatikan dulu

        print("Memulihkan stroke tipis dari binary awal...")
        gray = rescue_text_components(gray_raw_preserve, gray, border=4, min_area=3, dilate_keep=7)

        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.imshow(gray_raw_preserve, cmap='gray')
        plt.title("Binary awal sebelum cleaning")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(gray, cmap='gray')
        plt.title("Binary setelah cleaning + rescue")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        gray_before_despeckle = gray.copy()

        print("Menghapus speckle noise sambil menjaga stroke tipis...")
        gray = remove_speckle_noise_preserve_thin_strokes(
            gray, img,
            min_area=6,
            near_radius=9,
            band_pad=60,
            solidity_thr=0.22,
            body_area=220,
            body_h=18,
            small_keep_area=18,
            keep_dot_area=12,
            debug=True
        )

        print("Menghapus residual speckle yang masih lolos (body-aware)...")
        gray = remove_residual_speckles_by_body_proximity(
            gray,
            body_area=220,
            body_h=18,
            dist_keep=12,
            small_area_max=18,
            compact_fill_min=0.32,
            border=4,
            row_frac=0.002,
            pad_y=55,
            debug=True
        )

        print("Rescue akhir untuk menjaga struktur huruf...")
        gray = rescue_text_components(
            gray_before_despeckle,
            gray,
            border=4,
            min_area=4,
            dilate_keep=5
        )

print("Otsu T =", T, "Th =", Th)

plt.figure(figsize=(10,3))
plt.imshow(gray, cmap='gray')
plt.title("Binary yang dipakai (gray)")
plt.axis('off')
plt.show()

render = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.imshow(image_gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')

thresh = gray.copy()
kernel = np.ones((2,2), np.uint8)
opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=1)

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

moments = [np.zeros((1, 2)) for _ in range(num_slic)]
moments_void = [np.zeros((1, 2)) for _ in range(num_slic)]
for j in range(height):
    for i in range(width):
        if cue[j,i]!=0:
            moments[lbls[j,i]] = np.append(moments[lbls[j,i]], np.array([[i,j]]), axis=0)
            render[j,i,0]= 140-(10*(lbls[j,i]%6))
        else:
            moments_void[lbls[j,i]] = np.append(moments_void[lbls[j,i]], np.array([[i,j]]), axis=0)

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

scribe= nx.Graph()

filled=0
for n in range(num_slic):
    if ( len(moments[n])>SLIC_SPACE ):
        cx= int( np.mean( [array[0] for array in moments[n]] ))
        cy= int( np.mean( [array[1] for array in moments[n]] ))
        if (cue[cy,cx]!=0):
            render[cy,cx,1] = 255
            scribe.add_node(int(filled), label=int(lbls[cy,cx]), area=(len(moments[n])-1)/pow(SLIC_SPACE,2), hurf='', pos_bitmap=(cx,cy), pos_render=(cx,-cy), color='#FFA500', rasm=True)
            filled=filled+1

def pdistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

from dataclasses import dataclass, field
from typing import List
from typing import Optional

@dataclass
class ConnectedComponents:
    rect: (int,int,int,int)
    centroid: (int,int)
    area: Optional[int] = field(default=0)
    nodes: List[int] = field(default_factory=list)
    mat: Optional[np.ndarray] = field(default=None, repr=False)
    node_start: Optional[int] = field(default=-1)
    distance_start: Optional[int] = field(default=0)
    node_end: Optional[int] = field(default=-1)
    distance_end: Optional[int] = field(default=0)


pos = nx.get_node_attributes(scribe,'pos_bitmap')
components=[]
for n in range(scribe.number_of_nodes()):
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
        found=0
        for i in range(len(components)):
            if components[i].centroid==mc:
                components[i].nodes.append(n)
                tvane= freeman(seed[0]-mc[0], mc[1]-seed[1] )
                if seed[0]>mc[0] and pd>components[i].distance_start:
                    components[i].distance_start= pd
                    components[i].node_start= n
                elif seed[0]<mc[0] and pd>components[i].distance_end:
                    components[i].distance_end = pd
                    components[i].node_end= n
                found=1
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

components = sorted(components, key=lambda x: x.centroid[0], reverse=True)


# ====== SKELETON: ZHANG-SUEN HYBRID ======
# Semua metode thinning menggunakan Zhang-Suen (T.Y. Zhang & C.Y. Suen, 1984)
# "A Fast Parallel Algorithm for Thinning Digital Patterns"
# Communications of the ACM, 27(3), pp. 236-239, 1984.

skeleton_components_zhang     = []   # Zhang-Suen murni (langsung dari fg)
skeleton_components_mat       = []   # Zhang-Suen murni slot-2 (untuk pembanding, tanpa medial axis)
skeleton_components_dt        = []   # DT Ridge + Zhang-Suen thinning
skeleton_components_hybrid    = []   # Hybrid DT-core dengan Zhang-Suen (metode utama)
component_fg_masks            = []   # fg foreground ASLI tiap komponen (untuk masking blob)

# --- Estimasi baseline teks ---
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

component_types = []
local_blob_circs = []
local_blob_axes  = []

# --- Estimasi r_stroke global: median DT dari seluruh foreground ---
# r_stroke = jari-jari rata-rata stroke di seluruh gambar
# Digunakan sebagai satuan dasar threshold scalable (n * r^2)
_fg_global = (gray > 0)
if _fg_global.sum() > 0:
    _dt_global = distance_transform_edt(_fg_global)
    # Ambil hanya piksel foreground yang nilai DT-nya > 0
    _dt_vals = _dt_global[_fg_global]
    r_stroke_global = float(np.median(_dt_vals))
else:
    r_stroke_global = max(2.0, median_body_h / 8.0)
r_stroke_global = max(1.5, r_stroke_global)  # floor: minimal 1.5px
print(f"r_stroke_global (median DT foreground) = {r_stroke_global:.2f} px")

def _soft_gate_formula(val, center, width):
    return 1.0 / (1.0 + math.exp(-(val - center) / (width + 1e-9)))


def classify_formula_from_features(area_px, dy, circ, axis, bbox_ratio,
                                   r_stroke, median_body_h,
                                   mode="usulan",
                                   N_DIAC=math.pi/4.0,
                                   N_MAX=20.0*math.pi,
                                   fixed_diac_area=None,
                                   fixed_blob_area=None,
                                   extent=1.0,
                                   bw_bbox=1,
                                   bh_bbox=1):
    """
    mode:
      - usulan       : scalable + baseline + soft roundness
      - fixed_soft   : fixed threshold + baseline + soft roundness
      - area_only    : scalable + baseline, tanpa shape
      - hard_round   : scalable + baseline + hard roundness
    """
    r2 = float(r_stroke * r_stroke)
    dy_max = max(2.5 * float(r_stroke), 0.35 * float(median_body_h))

    if mode == "fixed_soft":
        area_diac_max = float(fixed_diac_area)
        area_blob_max = float(fixed_blob_area)
    else:
        area_diac_max = float(N_DIAC * r2)
        area_blob_max = float(N_MAX * r2)

    score_circ = _soft_gate_formula(circ,       center=0.40, width=0.08)
    score_axis = _soft_gate_formula(axis,       center=0.40, width=0.08)
    score_bbox = _soft_gate_formula(bbox_ratio, center=0.38, width=0.08)
    score_round = (score_circ + score_axis + score_bbox) / 3.0

    aspect = float(max(bw_bbox, bh_bbox) / (min(bw_bbox, bh_bbox) + 1e-9))
    stroke_veto = (
        ((aspect >= 2.4) and (bbox_ratio <= 0.42)) or
        ((circ <= 0.32) and (extent <= 0.20) and (bbox_ratio <= 0.55))
    )

    if area_px < area_diac_max and dy > dy_max:
        label = "diacritic"
    else:
        round_ok = False
        if area_px <= area_blob_max:
            if mode in ("usulan", "fixed_soft"):
                round_ok = (score_round >= 0.50)
            elif mode == "hard_round":
                round_ok = (circ >= 0.40 and axis >= 0.40 and bbox_ratio >= 0.38)
            elif mode == "area_only":
                round_ok = True

        if round_ok and (not stroke_veto) and dy <= (2.0 * dy_max):
            label = "circular_blob"
        else:
            label = "normal"

    return {
        "label": label,
        "score_round": float(score_round),
        "area_diac_max": float(area_diac_max),
        "area_blob_max": float(area_blob_max),
        "dy_max": float(dy_max),
    }


def extract_raw_components_for_formula(gray, baseline_y):
    """
    Ambil raw connected components langsung dari binary gray.
    Ini sengaja dipakai untuk analisis rumus agar jumlah kandidat komponen
    lebih banyak dan pembanding lebih informatif.
    """
    bw = (gray > 0).astype(np.uint8)
    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)

    rows = []
    for i in range(1, n):
        x, y, w, h, area_stat = stats[i]
        if area_stat < 6:
            continue

        comp_mask = (labels == i).astype(np.uint8) * 255
        area_px = int(cv.countNonZero(comp_mask))
        area_cnt, per, circ, axis = _contour_metrics(comp_mask)
        bbox_ratio, extent, bw_box, bh_box = _bbox_extent_roundness(comp_mask)

        cx, cy = cents[i]
        dy = abs(float(cy) - float(baseline_y))

        rows.append({
            "idx": int(i),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": float(cx), "cy": float(cy),
            "area_px": int(area_px),
            "dy": float(dy),
            "circ": float(circ),
            "axis": float(axis),
            "bbox_ratio": float(bbox_ratio),
            "extent": float(extent),
        })

    return rows


def _write_csv_formula(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _draw_overlay_formula(gray, rows, out_path, title_text):
    canvas = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    color_map = {
        "diacritic": (0, 0, 255),       # merah
        "circular_blob": (0, 180, 0),   # hijau
        "normal": (255, 0, 0),          # biru
    }

    for r in rows:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cx, cy = int(round(r["cx"])), int(round(r["cy"]))
        lab = r["label"]
        color = color_map.get(lab, (255, 255, 255))

        cv.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        cv.putText(canvas, lab[0].upper(), (x, max(12, y - 2)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv.LINE_AA)
        cv.circle(canvas, (cx, cy), 1, color, -1)

    plt.figure(figsize=(12, 4))
    plt.imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
    plt.title(title_text)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_threshold_scaling_formula(imagename, r_stroke_global):
    scales = np.linspace(0.5, 1.6, 200)

    fixed_diac = (math.pi / 4.0) * (r_stroke_global ** 2)
    fixed_blob = (20.0 * math.pi) * (r_stroke_global ** 2)

    scalable_diac = (math.pi / 4.0) * ((r_stroke_global * scales) ** 2)
    scalable_blob = (20.0 * math.pi) * ((r_stroke_global * scales) ** 2)

    plt.figure(figsize=(8, 5))
    plt.plot(scales, scalable_diac, label=r"Usulan: $\pi/4 \cdot (sr)^2$")
    plt.plot(scales, scalable_blob, label=r"Usulan: $20\pi \cdot (sr)^2$")
    plt.plot(scales, np.full_like(scales, fixed_diac), "--", label=f"Fixed diac = {fixed_diac:.1f}")
    plt.plot(scales, np.full_like(scales, fixed_blob), "--", label=f"Fixed blob = {fixed_blob:.1f}")
    plt.axvline(1.0, linestyle=":", label="Skala referensi = 1.0")
    plt.xlabel("Faktor skala s")
    plt.ylabel("Threshold area")
    plt.title("Mengapa rumus scalable lebih stabil terhadap perubahan skala")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{imagename}_proof_threshold_scaling.png", dpi=200)
    plt.close()


def _plot_stability_formula(imagename, summary_rows):
    methods = ["usulan", "fixed_soft", "area_only", "hard_round"]
    labels = {
        "usulan": "Rumus usulan",
        "fixed_soft": "Threshold tetap",
        "area_only": "Area-only",
        "hard_round": "Hard-roundness",
    }

    scales = sorted(set(float(r["scale"]) for r in summary_rows))
    plt.figure(figsize=(8, 5))

    for m in methods:
        ys = []
        for s in scales:
            rec = [r for r in summary_rows if r["method"] == m and float(r["scale"]) == s]
            ys.append(rec[0]["stability_vs_1x"] if rec else 0.0)
        plt.plot(scales, ys, marker="o", label=labels[m])

    plt.ylim(0, 1.05)
    plt.xlabel("Faktor skala")
    plt.ylabel("Stabilitas label vs skala 1.0")
    plt.title("Stabilitas klasifikasi komponen saat skala berubah")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{imagename}_proof_label_stability.png", dpi=200)
    plt.close()


def _plot_counts_formula(imagename, summary_rows):
    methods = ["usulan", "fixed_soft", "area_only", "hard_round"]
    labels = {
        "usulan": "Rumus usulan",
        "fixed_soft": "Threshold tetap",
        "area_only": "Area-only",
        "hard_round": "Hard-roundness",
    }

    recs = [r for r in summary_rows if float(r["scale"]) == 1.0]

    diac = [next(rr["count_diacritic"] for rr in recs if rr["method"] == m) for m in methods]
    circ = [next(rr["count_circular_blob"] for rr in recs if rr["method"] == m) for m in methods]
    norm = [next(rr["count_normal"] for rr in recs if rr["method"] == m) for m in methods]

    x = np.arange(len(methods))
    plt.figure(figsize=(9, 5))
    plt.bar(x, diac, label="diacritic")
    plt.bar(x, circ, bottom=diac, label="circular_blob")
    plt.bar(x, norm, bottom=np.array(diac) + np.array(circ), label="normal")
    plt.xticks(x, [labels[m] for m in methods], rotation=15)
    plt.ylabel("Jumlah komponen")
    plt.title("Jumlah komponen per kelas pada skala 1.0")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{imagename}_proof_counts_1x.png", dpi=200)
    plt.close()


def run_formula_proof_package(gray, baseline_y, median_body_h, r_stroke_global, imagename):
    raw_feats = extract_raw_components_for_formula(gray, baseline_y)

    if len(raw_feats) == 0:
        print("[FORMULA PROOF] Tidak ada raw components yang bisa dianalisis.")
        return

    # Fixed threshold dibuat adil: sama dengan rumus usulan pada skala 1.0
    fixed_diac_area = (math.pi / 4.0) * (r_stroke_global ** 2)
    fixed_blob_area = (20.0 * math.pi) * (r_stroke_global ** 2)

    scales = [0.70, 1.00, 1.40]
    methods = ["usulan", "fixed_soft", "area_only", "hard_round"]

    detail_rows = []

    for s in scales:
        r_s = r_stroke_global * s
        for f in raw_feats:
            area_s = float(f["area_px"]) * (s ** 2)
            dy_s = float(f["dy"]) * s

            for method in methods:
                out = classify_formula_from_features(
                    area_px=area_s,
                    dy=dy_s,
                    circ=f["circ"],
                    axis=f["axis"],
                    bbox_ratio=f["bbox_ratio"],
                    r_stroke=r_s,
                    median_body_h=median_body_h * s,
                    mode=method,
                    fixed_diac_area=fixed_diac_area,
                    fixed_blob_area=fixed_blob_area,
                    extent=f["extent"],
                    bw_bbox=f["w"],
                    bh_bbox=f["h"],
                )

                detail_rows.append({
                    "idx": f["idx"],
                    "scale": float(s),
                    "method": method,
                    "x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"],
                    "cx": f["cx"], "cy": f["cy"],
                    "area_px_original": f["area_px"],
                    "area_px_scaled": float(area_s),
                    "dy_original": f["dy"],
                    "dy_scaled": float(dy_s),
                    "circ": f["circ"],
                    "axis": f["axis"],
                    "bbox_ratio": f["bbox_ratio"],
                    "extent": f["extent"],
                    "r_stroke_scaled": float(r_s),
                    "score_round": out["score_round"],
                    "area_diac_max": out["area_diac_max"],
                    "area_blob_max": out["area_blob_max"],
                    "dy_max": out["dy_max"],
                    "label": out["label"],
                })

    # Summary per method per scale
    summary_rows = []
    label_map = defaultdict(dict)

    for r in detail_rows:
        key = (r["method"], float(r["scale"]))
        label_map[key][r["idx"]] = r["label"]

    for method in methods:
        base = label_map[(method, 1.0)]
        for s in scales:
            cur = label_map[(method, float(s))]
            same = sum(1 for k in base if cur.get(k) == base[k])
            stability = same / max(1, len(base))

            counts = Counter(cur.values())
            summary_rows.append({
                "method": method,
                "scale": float(s),
                "num_components": int(len(cur)),
                "count_diacritic": int(counts.get("diacritic", 0)),
                "count_circular_blob": int(counts.get("circular_blob", 0)),
                "count_normal": int(counts.get("normal", 0)),
                "stability_vs_1x": float(stability),
            })

    # CSV
    _write_csv_formula(
        f"{imagename}_proof_formula_detail.csv",
        detail_rows,
        list(detail_rows[0].keys())
    )

    _write_csv_formula(
        f"{imagename}_proof_formula_summary.csv",
        summary_rows,
        list(summary_rows[0].keys())
    )

    # Overlay 1x usulan
    rows_usulan_1x = [r for r in detail_rows if r["method"] == "usulan" and float(r["scale"]) == 1.0]
    rows_fixed_1x  = [r for r in detail_rows if r["method"] == "fixed_soft" and float(r["scale"]) == 1.0]

    _draw_overlay_formula(
        gray, rows_usulan_1x,
        f"{imagename}_proof_overlay_usulan_1x.png",
        "Overlay label komponen - Rumus usulan (skala 1.0)"
    )

    _draw_overlay_formula(
        gray, rows_fixed_1x,
        f"{imagename}_proof_overlay_fixed_1x.png",
        "Overlay label komponen - Threshold tetap (skala 1.0)"
    )

    _plot_threshold_scaling_formula(imagename, r_stroke_global)
    _plot_stability_formula(imagename, summary_rows)
    _plot_counts_formula(imagename, summary_rows)

    print("\n[FORMULA PROOF] Output dibuat:")
    print(f"  - {imagename}_proof_formula_detail.csv")
    print(f"  - {imagename}_proof_formula_summary.csv")
    print(f"  - {imagename}_proof_overlay_usulan_1x.png")
    print(f"  - {imagename}_proof_overlay_fixed_1x.png")
    print(f"  - {imagename}_proof_threshold_scaling.png")
    print(f"  - {imagename}_proof_label_stability.png")
    print(f"  - {imagename}_proof_counts_1x.png")


run_ablation_formula_vs_constants(
    components=components,
    baseline_y=baseline_y,
    median_body_h=median_body_h,
    r_stroke_global=r_stroke_global,
    gray_image=gray,
    imagename=imagename
)

run_formula_proof_package(
    gray=gray,
    baseline_y=baseline_y,
    median_body_h=median_body_h,
    r_stroke_global=r_stroke_global,
    imagename=imagename
)



if len(components) == 0:
    print("WARNING: Tidak ada komponen yang ditemukan untuk skeletonisasi!")
    combined_skeleton = np.zeros((height, width), dtype=np.uint8)
else:
    print(f"\n{'='*60}")
    print(f"MEMPROSES {len(components)} KOMPONEN")
    print(f"Metode Skeleton: ZHANG-SUEN THINNING MURNI (Zhang & Suen, 1984) — tanpa medial axis")
    print(f"Baseline Y: {baseline_y:.1f}, Median Body Height: {median_body_h:.1f}")
    print(f"{'='*60}\n")

# ← LOOP PEMBUATAN SKELETON ZHANG-SUEN DENGAN HYBRID STRUCTURE
for n in range(len(components)):
    k_sk  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    k_dot = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))

    binary_mat = ((components[n].mat == RASMVAL).astype(np.uint8) * 255)
    holes = get_holes_mask(binary_mat)

    # fg_orig: foreground ASLI blob ini, sebelum morphologi apapun
    # Dipakai nanti untuk masking area circular blob pada combined_skeleton
    fg_orig = (binary_mat > 0)

    area_px = cv.countNonZero(binary_mat)
    cy = components[n].centroid[1]

    blob_type, circ, axis, dy = detect_blob_type(binary_mat, cy, baseline_y, median_body_h, r_stroke=r_stroke_global)
    component_types.append(blob_type)

    _r2 = r_stroke_global ** 2
    print(f"Komponen {n}: type={blob_type:15s} | area={area_px:4d} "
          f"(diac<{math.pi/4*_r2:.0f}, blob<{20*math.pi*_r2:.0f}) | "
          f"r_stroke={r_stroke_global:.2f}px | circ={circ:.3f} | axis={axis:.3f} | dy={dy:.1f}")

    if area_px >= 1200:
        binary_mat = cv.morphologyEx(binary_mat, cv.MORPH_CLOSE, k_sk, iterations=1)
    else:
        binary_mat = cv.morphologyEx(binary_mat, cv.MORPH_OPEN, k_dot, iterations=1)

    binary_mat[holes == 255] = 0

    _k_erosi = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    _fg_eroded = cv.erode(binary_mat, _k_erosi, iterations=1)
    _fg_eroded[holes == 255] = 0
    fg = (_fg_eroded > 0)

    fg_used = fg
    forced_loop = False
    cores = []

    # ===== Baseline: Zhang-Suen murni (untuk pembanding DT ridge) =====
    # ← ZHANG: skeletonize langsung dari fg menggunakan Zhang-Suen
    skel_zhang_pure = zhang_suen_thinning(fg_used).astype(np.uint8)

    # ===== Zhang-Suen (pembanding, dipakai di combined_mat slot) =====
    # Tidak ada medial axis — slot skel_mat juga diisi Zhang-Suen murni
    skel_mat = zhang_suen_thinning(fg_used).astype(np.uint8)

    # ===== DT Ridge + Zhang-Suen thinning =====
    # ← ZHANG: dt_ridge_skeleton_zhang menggunakan Zhang-Suen internally
    skel_dt, _ = dt_ridge_skeleton_zhang(fg_used, r_min=1.0, neigh=3)
    skel_dt = (skel_dt > 0).astype(np.uint8)


    # ← STRATEGI HYBRID PER TIPE KOMPONEN (semua thinning = Zhang-Suen)

    if blob_type == 'diacritic':
        # DIAKRITIK: Zhang-Suen murni langsung dari fg
        # ← ZHANG: tidak ada medial axis, langsung Zhang-Suen
        skel_hybrid = zhang_suen_thinning(fg_used).astype(np.uint8)
        print(f"  → Skeleton Zhang-Suen murni (diakritik)")

    elif blob_type == 'circular_blob':
        # GUMPALAN BULAT: circular skeleton + area tengah wajib kosong
        print(f"  → Skeleton Zhang-Suen: CIRCULAR (gumpalan bulat, hollow center)")
    
        if np.count_nonzero(holes) == 0:
            fg_used, forced_loop = carve_pseudo_holes_by_dt_peaks(
                fg,
                peak_neigh=5,
                r_peak_min=2.0,
                alpha=0.65,
                circ_min=0.35,
                axis_min=0.35,
                max_carves=2,
                min_core_area=12,
                extent_min=0.20,
                bbox_min=0.30,
                core_ratio_max=0.80,
                max_len_vs_r0=8.0
            )
            if forced_loop:
                print(f"  → Pseudo-hole berhasil di-carve!")
    
        # EROSI 1px pada fg sebelum skeleton circular
        # Tujuan: haluskan tepi blob agar kontur skeleton lebih bersih
        _k_circ_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        _fg_circ_eroded_u8 = cv.erode(
            (fg_used.astype(np.uint8) * 255), _k_circ_erode, iterations=1
        )
        _fg_circ_eroded_u8[holes == 255] = 0   # jaga holes tetap ada
        fg_for_circ = (_fg_circ_eroded_u8 > 0)
    
        # ← ZHANG: create_circular_skeleton_zhang menggunakan Zhang-Suen
        # Input: fg SETELAH erosi 1px
        skel_circular = create_circular_skeleton_zhang(
            fg_for_circ, min_radius=2.0, dt_percentile=60, force_circular=True
        )
            
        # Hanya hapus garis di dalam lingkaran lokal.
        # Bentuk skeleton luar tetap dibiarkan apa adanya.
        local_core_mask = get_local_circular_core_mask(
            fg_for_circ, min_radius=2.0, dt_percentile=60
        )
        
        skel_circular = remove_inner_line_only(skel_circular, local_core_mask)
        
        # Skeleton lain juga dibersihkan di area dalam lingkaran lokal
        skel_zhang_pure = remove_inner_line_only(skel_zhang_pure, local_core_mask)
        skel_mat        = remove_inner_line_only(skel_mat, local_core_mask)
        skel_dt         = remove_inner_line_only(skel_dt, local_core_mask)
        
        print(f"  → Hanya garis di dalam lingkaran lokal yang dihapus")
                
        
        
        # Mask isi blob supaya skeleton non-circular tidak bisa masuk ke tengah blob.
        _filled_blob_mask, _ring_blob_mask = _largest_filled_contour_mask(fg_for_circ)
        _interior_mask = _filled_blob_mask & (~_ring_blob_mask)
        skel_zhang_pure[_interior_mask] = 0
        skel_mat[_interior_mask]        = 0
        skel_dt[_interior_mask]         = 0
        print(f"  → Skeleton lain yang masuk ke area dalam circular blob di-mask 0")
    
        if np.any(skel_circular > 0):
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1)
            plt.imshow(binary_mat, cmap='gray')
            plt.title(f"Comp {n}: Binary Input")
            plt.axis('off')
            plt.subplot(1, 4, 2)
            plt.imshow(fg_used, cmap='gray')
            plt.title(f"FG after carve (carved={forced_loop})")
            plt.axis('off')
            plt.subplot(1, 4, 3)
            plt.imshow(fg_for_circ, cmap='gray')
            plt.title(f"FG after 1px erosion")
            plt.axis('off')
            plt.subplot(1, 4, 4)
            plt.imshow(skel_circular, cmap='gray')
            plt.title(f"Circular Skeleton HOLLOW\n(circ={circ:.2f})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
        skel_hybrid = skel_circular

    else:  # 'normal'
        # HURUF NORMAL: Hybrid DT-core dengan Zhang-Suen thinning murni
        _fg_total_area = float(fg.sum())
        fg_used, forced_loop, cores = carve_pseudo_holes_by_dt_peaks(
            fg,
            peak_neigh=7,
            r_peak_min=2.0,
            alpha=0.72,
            circ_min=0.38,
            axis_min=0.38,
            extent_min=0.25,
            bbox_min=0.30,
            core_ratio_max=0.80,
            max_len_vs_r0=5.0,
            max_carves=12,
            min_core_area=12,
            return_cores=True,
            debug=True
        )

        if len(cores) > 0:
            _filtered = []
            for _cm in cores:
                _cm_frac = float(_cm.sum()) / (_fg_total_area + 1e-9)
                if _cm_frac <= 0.40:
                    _filtered.append(_cm)
                else:
                    print(f"  SKIP core frac={_cm_frac:.2f} (terlalu besar)")
            cores = _filtered

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

        # ← ZHANG: skeleton fg ASLI murni Zhang-Suen (tanpa medial axis)
        skel_thin_orig = zhang_suen_thinning(fg).astype(np.uint8)

        if len(cores) > 0:
            cores_mask = np.zeros(fg.shape, dtype=bool)
            for core_mask in cores:
                core_dil = cv.dilate(
                    (core_mask.astype(np.uint8)*255),
                    cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations=2)
                cores_mask |= (core_dil > 0)

            # ← ZHANG: skeleton fg_used (berlubang) murni Zhang-Suen (tanpa medial axis)
            skel_thin_carved = zhang_suen_thinning(fg_used).astype(np.uint8)
            skel_core_only = (skel_thin_carved > 0) & cores_mask

            # ← ZHANG: Gabungan ditipiskan dengan Zhang-Suen
            skel_hybrid = zhang_suen_thinning(
                (skel_thin_orig > 0) | skel_core_only
            ).astype(np.uint8)
        else:
            skel_hybrid = skel_thin_orig

        # ← ZHANG: pastikan 1-pixel dengan Zhang-Suen
        skel_hybrid = zhang_suen_thinning(skel_hybrid > 0).astype(np.uint8)
        print(f"  → Skeleton Zhang-Suen murni: Hybrid DT-core (normal)")

    # ================================================================
    # DT-LOOP:
    # Thinning loop menggunakan Zhang-Suen (1984)
    # ================================================================
skel_hybrid_before_loop = skel_hybrid.copy()
if blob_type == 'circular_blob':
    loop_added = np.zeros_like(skel_hybrid, dtype=np.uint8)
    thick_reg = np.zeros(binary_mat.shape, dtype=bool)
    skel_hybrid = enforce_hollow_circular_blob(skel_hybrid, fg_for_circ)
    print(f"  → [DT-LOOP Zhang] SKIP untuk circular_blob agar tengah tetap kosong")
else:
    skel_hybrid, loop_added, thick_reg = develop_loop_from_threshold(
        fg_bool   = (binary_mat > 0),
        skel_u8   = skel_hybrid,
        dt_loop_threshold = DT_LOOP_THRESHOLD
    )
    skel_hybrid = (skel_hybrid > 0).astype(np.uint8)

# ============================================================
# CLEANUP FINAL KOMPONEN:
# hapus garis-garis yang berada di dalam loop tertutup
# ============================================================
skel_hybrid = remove_inner_lines_from_closed_loops(
    skel_hybrid,
    min_loop_len=8,
    min_loop_area=12
)

n_loop_px = int(loop_added.sum())

if n_loop_px > 0:
    print(
        f"  → [DT-LOOP Zhang] Lokasi tebal terdeteksi (DT>{DT_LOOP_THRESHOLD}px): "
        f"{int(thick_reg.sum())} px tebal | {n_loop_px} px loop ditambahkan"
    )

    fig_loop, axes_loop = plt.subplots(1, 4, figsize=(16, 4))

    axes_loop[0].imshow(binary_mat, cmap='gray')
    axes_loop[0].set_title(f"Comp {n}: Binary Threshold Asli")
    axes_loop[0].axis('off')

    from scipy.ndimage import distance_transform_edt as edt_scipy
    _dt_local = edt_scipy((binary_mat > 0).astype(bool))
    im_dt_local = axes_loop[1].imshow(_dt_local, cmap='viridis')
    axes_loop[1].set_title(f"Distance Transform\n(threshold={DT_LOOP_THRESHOLD}px)")
    fig_loop.colorbar(im_dt_local, ax=axes_loop[1], label="Distance")

    axes_loop[2].imshow(skel_hybrid_before_loop, cmap='gray')
    _ovl = np.zeros((*skel_hybrid_before_loop.shape, 3), dtype=np.uint8)
    _ovl[skel_hybrid_before_loop > 0] = [255, 255, 255]
    _ovl[thick_reg] = [255, 80, 80]
    axes_loop[2].imshow(_ovl, alpha=0.7)
    axes_loop[2].set_title(
        f"Skeleton Zhang + Area Tebal (merah)\nDT>{DT_LOOP_THRESHOLD}px"
    )
    axes_loop[2].axis('off')

    axes_loop[3].imshow(skel_hybrid, cmap='gray')
    _ovl2 = np.zeros((*skel_hybrid.shape, 3), dtype=np.uint8)
    _ovl2[skel_hybrid > 0] = [255, 255, 255]
    _ovl2[loop_added > 0] = [80, 255, 80]
    axes_loop[3].imshow(_ovl2, alpha=0.7)
    axes_loop[3].set_title("Skeleton Zhang Final\n(hijau=loop dari threshold asal)")
    axes_loop[3].axis('off')

    plt.suptitle(
        f"Komponen {n} [{blob_type}] — Zhang-Suen DT-Loop Development",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()

else:
    print(
        f"  → [DT-LOOP Zhang] Tidak ada lokasi tebal "
        f"(DT>{DT_LOOP_THRESHOLD}px) → skeleton tetap"
    )

skeleton_components_zhang.append(skel_zhang_pure)
skeleton_components_mat.append(skel_mat)     # slot ini juga Zhang-Suen murni
skeleton_components_dt.append(skel_dt)
skeleton_components_hybrid.append(skel_hybrid)
component_fg_masks.append(fg_orig)           # fg asli komponen ini (untuk masking blob)

print(f"\n{'='*60}\n")

print(f"\n{'='*60}")
print("RINGKASAN DETEKSI TIPE KOMPONEN:")
print(f"Metode Skeleton: ZHANG-SUEN THINNING MURNI (Zhang & Suen, 1984) — tanpa medial axis")
print(f"{'='*60}")
print(f"Total komponen: {len(component_types)}")
print(f"- Diacritic:     {component_types.count('diacritic')}")
print(f"- Circular Blob: {component_types.count('circular_blob')}")
print(f"- Circular Blob (lokal DT-core): {len(local_blob_circs)}")
if len(local_blob_circs) > 0:
    print(f"  circ_local avg/min/max: {np.mean(local_blob_circs):.3f}/"
          f"{np.min(local_blob_circs):.3f}/{np.max(local_blob_circs):.3f}")
print(f"- Normal:        {component_types.count('normal')}")
print(f"{'='*60}\n")

# Gabungkan semua skeleton dengan circular blob masking
H, W = height, width

# Buat mask blob: union semua area circular_blob (dengan sedikit dilate)
# Ini adalah "tembok" — skeleton dari luar tidak boleh masuk ke dalamnya
blob_wall = np.zeros((H, W), dtype=bool)
for _ctype, _fgmask in zip(component_types, component_fg_masks):
    if _ctype == 'circular_blob':
        blob_wall |= (_fgmask > 0)

print("Menggabungkan skeleton dengan circular blob masking...")

combined_hybrid = np.zeros((H, W), dtype=np.uint8)

# Pass 1: tambahkan semua komponen NON-circular-blob
# tapi POTONG pikselnya jika masuk ke dalam area blob_wall
for _skel, _ctype in zip(skeleton_components_hybrid, component_types):
    if _ctype == 'circular_blob':
        continue
    # potong garis yang masuk ke dalam area blob
    _skel_clipped = (_skel > 0) & (~blob_wall)
    combined_hybrid |= _skel_clipped.astype(np.uint8)

# Pass 2: tambahkan skel_circular tiap blob (tanpa clipping)
for _skel, _ctype in zip(skeleton_components_hybrid, component_types):
    if _ctype != 'circular_blob':
        continue
    combined_hybrid |= (_skel > 0).astype(np.uint8)

combined_hybrid = (combined_hybrid > 0).astype(np.uint8) * 255

combined_hybrid = remove_inner_lines_from_closed_loops(
    combined_hybrid,
    min_loop_len=8,
    min_loop_area=12
)
combined_hybrid = (combined_hybrid > 0).astype(np.uint8) * 255

# combined_zhang / mat / dt: gabung biasa (untuk pembanding)
combined_zhang = combine_component_skeletons(skeleton_components_zhang, (H, W)) if skeleton_components_zhang else np.zeros((H, W), np.uint8)
combined_mat   = combine_component_skeletons(skeleton_components_mat,   (H, W)) if skeleton_components_mat   else np.zeros((H, W), np.uint8)
combined_dt    = combine_component_skeletons(skeleton_components_dt,    (H, W)) if skeleton_components_dt    else np.zeros((H, W), np.uint8)

combined_skeleton = combined_hybrid.copy()
print("Penggabungan & masking selesai.")



# TEPI & DT
print("Menghitung kontur (boundary) & DT (jarak ke tepi terdekat)...")
bin255 = ((gray > 0).astype(np.uint8) * 255)
contours, hierarchy = cv.findContours(bin255, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
contour_img = np.zeros_like(bin255, dtype=np.uint8)
if len(contours) > 0:
    cv.drawContours(contour_img, contours, -1, 255, 1)
src = np.full_like(contour_img, 255, dtype=np.uint8)
src[contour_img > 0] = 0
dt = cv.distanceTransform(src, cv.DIST_L2, 5)
dt_edge = dt

from scipy.ndimage import distance_transform_edt as edt_scipy
print("\nMenampilkan Distance Transform (gaya dosen)...")
_bin_bool_viz = (gray > 0).astype(bool)
dt_viz = edt_scipy(_bin_bool_viz)

with open(f"{imagename}_dt_pixels.txt", "w") as f_dt:
    for i in range(dt_viz.shape[0]):
        for j in range(dt_viz.shape[1]):
            val = dt_viz[i,j]
            line = f"Pixel ({i},{j}) = {val}"
            print(line)
            f_dt.write(line + "\n")
print(f"\nSemua nilai pixel DT tersimpan di: {imagename}_dt_pixels.txt")

fig_dt, axes_dt = plt.subplots(1, 2, figsize=(12, 5))
axes_dt[0].imshow(gray, cmap='gray')
axes_dt[0].set_title("Original Image")
im_dt = axes_dt[1].imshow(dt_viz, cmap='viridis')
axes_dt[1].set_title("Distance Transform")
fig_dt.colorbar(im_dt, ax=axes_dt[1], label="Distance")
plt.tight_layout()
plt.show()

edges = (contour_img > 0).astype(np.uint8)


# Plot DT pada skeleton untuk semua metode
methods = {
    "Zhang-Suen (murni)":              combined_zhang,
    "Zhang-Suen (slot-2, murni)":      combined_mat,
    "DT Ridge + Zhang-Suen":           combined_dt,
    "Hybrid DT-core + Zhang-Suen":     combined_hybrid,
}

for name, sk in methods.items():
    dt_skel_map, dist_vals = dt_map_at_skeleton(sk, dt_edge)
    if dist_vals.size > 0:
        print(f"[{name}] DT(skel->edge) mean/median/min/max:",
              float(dist_vals.mean()),
              float(np.median(dist_vals)),
              float(dist_vals.min()),
              float(dist_vals.max()))
    else:
        print(f"[{name}] skeleton kosong.")

    plt.figure(figsize=(10, 3))
    plt.imshow(np.zeros_like(dt_edge), cmap="gray", vmin=0, vmax=1)
    alpha = (dt_skel_map > 0).astype(np.float32)
    plt.imshow(dt_skel_map, cmap="jet", alpha=alpha)
    contour_rgba = np.zeros((*contour_img.shape, 4), dtype=np.float32)
    contour_rgba[contour_img > 0] = [1.0, 1.0, 1.0, 1.0]
    plt.imshow(contour_rgba)
    plt.imshow(contour_img, cmap="gray", alpha=0.35)
    plt.title(f"DT on Skeleton | {name}")
    plt.axis("off")
    plt.show()


# EVALUASI
def evaluasi_skeleton_lengkap(skeleton, edge, dt_edge, component_types):
    panjang = np.sum(skeleton > 0)
    overlap = np.logical_and(skeleton, edge).sum()
    num_labels, labels = cv.connectedComponents(skeleton.astype(np.uint8))
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

if combined_skeleton is not None and np.any(combined_skeleton > 0):
    plt.figure(figsize=(8, 6))
    plt.imshow(combined_skeleton, cmap='gray')
    plt.title("Hybrid Skeleton Zhang-Suen \nDistance Transform + Circular Blob Detection")
    plt.axis('off')
    plt.show()

    hasil = evaluasi_skeleton_lengkap(combined_skeleton, edges, dt_edge, component_types)
    print("\n" + "="*60)
    print("EVALUASI SKELETON ZHANG-SUEN")
    print("Algoritma: T.Y. Zhang & C.Y. Suen, CACM 27(3), 1984")
    print("="*60)
    for k, v in hasil.items():
        if isinstance(v, float):
            print(f"{k:30s}: {v:.4f}")
        else:
            print(f"{k:30s}: {v}")
    print("="*60 + "\n")
else:
    print("Skeleton kosong, tidak ada data untuk ditampilkan.")


# Visualisasi tambahan
if combined_skeleton is not None and np.any(combined_skeleton > 0):
    dist_on_skel = dt[combined_skeleton > 0]
    if dist_on_skel.size > 0:
        print("DT(skeleton->contour) mean/median/min/max:",
              float(dist_on_skel.mean()),
              float(np.median(dist_on_skel)),
              float(dist_on_skel.min()),
              float(dist_on_skel.max()))

plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.show()

plt.imshow(contour_img, cmap="gray")
plt.title("Contour Image (1px)")
plt.show()

plt.figure(figsize=(10,3))
plt.imshow(dt, cmap='jet')
plt.imshow(combined_skeleton>0, cmap='gray', alpha=0.6)
plt.title("DT map + Zhang-Suen Skeleton overlay\n(Zhang & Suen, 1984)")
plt.axis('off')
plt.show()


# =============================================================
# SIG CURVE RECONSTRUCTION DARI SKELETON ZHANG-SUEN
# =============================================================
from scipy.spatial import KDTree

def build_skeleton_pixel_graph(skel_mask):
    ys, xs = np.where(skel_mask > 0)
    coords = list(zip(xs.tolist(), ys.tolist()))
    coord_set = set(coords)
    G = nx.Graph()
    for (x, y) in coords:
        G.add_node((x, y))
    for (x, y) in coords:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb in coord_set:
                    w = math.sqrt(dx*dx + dy*dy)
                    G.add_edge((x, y), nb, weight=w)
    return G, coords


def skeleton_graph_to_ordered_path(G, coords):
    if len(coords) == 0:
        return []
    coord_set = set(coords)
    adj = {c: [] for c in coords}
    for (x, y) in coords:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb in coord_set:
                    adj[(x, y)].append(nb)
    visited_global = set()
    components = []
    for seed in coords:
        if seed in visited_global:
            continue
        queue = [seed]
        comp = []
        visited_global.add(seed)
        head = 0
        while head < len(queue):
            cur = queue[head]; head += 1
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited_global:
                    visited_global.add(nb)
                    queue.append(nb)
        components.append(comp)
    components.sort(key=lambda c: min(p[0] for p in c))
    result = []
    for comp in components:
        if not comp:
            continue
        comp_set = set(comp)
        degree = {p: len([nb for nb in adj[p] if nb in comp_set]) for p in comp}
        endpoints = [p for p in comp if degree[p] == 1]
        if endpoints:
            start = min(endpoints, key=lambda p: (p[0], p[1]))
        else:
            start = min(comp, key=lambda p: (p[0], p[1]))
        adj_sorted = {}
        for p in comp:
            nbs = [nb for nb in adj[p] if nb in comp_set]
            adj_sorted[p] = sorted(nbs, key=lambda nb: (nb[0], -nb[1]), reverse=True)
        primary_visited = set()
        order = []
        stack = [(start, 0)]
        primary_visited.add(start)
        order.append(start)
        while stack:
            cur, idx = stack[-1]
            nbs = adj_sorted[cur]
            found = False
            while idx < len(nbs):
                nb = nbs[idx]
                idx += 1
                if nb not in primary_visited:
                    stack[-1] = (cur, idx)
                    primary_visited.add(nb)
                    order.append(nb)
                    stack.append((nb, 0))
                    found = True
                    break
            if not found:
                stack.pop()
                if stack:
                    backtrack_node = stack[-1][0]
                    has_unvisited_branch = any(
                        nb not in primary_visited
                        for nb in adj_sorted[backtrack_node]
                    )
                    if has_unvisited_branch:
                        order.append(backtrack_node)
        missing = [p for p in comp if p not in primary_visited]
        if missing:
            missing.sort(key=lambda p: (p[0], p[1]))
            order.extend(missing)
        result.extend(order)
    return result


def gaussian_smooth_curve(points_xy, sigma=1.5):
    pts = np.array(points_xy, dtype=float)
    if len(pts) < 5:
        return pts
    kernel_size = max(3, int(6 * sigma) | 1)
    kernel_size = min(kernel_size, len(pts) // 2 * 2 - 1)
    if kernel_size < 3:
        return pts
    x_kern = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-(x_kern**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    smoothed = pts.copy()
    for dim in range(2):
        padded = np.pad(pts[:, dim], kernel_size // 2, mode='edge')
        smoothed[:, dim] = np.convolve(padded, kernel, mode='valid')
    return smoothed


class SIGCurveReconstruction:
    """
    SIG Curve Reconstruction dari skeleton Zhang-Suen.
    Skeleton dibuat dengan algoritma Zhang-Suen (T.Y. Zhang & C.Y. Suen, 1984).
    """
    def __init__(self, sigma=1.5):
        self.sigma = sigma
        self.points = None
        self.ordered_points = None
        self.curve = None

    def fit_from_skeleton_mask(self, skel_mask):
        G, coords = build_skeleton_pixel_graph(skel_mask)
        if len(coords) == 0:
            self.curve = np.empty((0, 2))
            return self.curve
        self.points = np.array(coords, dtype=float)
        ordered = skeleton_graph_to_ordered_path(G, coords)
        self.ordered_points = np.array(ordered, dtype=float)
        self.curve = gaussian_smooth_curve(self.ordered_points, sigma=self.sigma)
        return self.curve

    def plot_reconstruction(self, ax=None, show_original=True,
                            title='SIG Curve Reconstruction (Zhang-Suen)'):
        if self.curve is None:
            raise ValueError("Panggil fit_from_skeleton_mask() dulu.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        if show_original and self.points is not None:
            ax.scatter(self.points[:, 0], self.points[:, 1],
                       c='blue', s=8, alpha=0.5, label='Zhang-Suen skeleton')
        if len(self.curve) > 1:
            ax.plot(self.curve[:, 0], self.curve[:, 1],
                    'r-', linewidth=1.5, label='Reconstructed curve')
        ax.scatter(self.curve[:, 0], self.curve[:, 1],
                   c='red', s=15, zorder=5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=8)
        return ax


# SIG per komponen
print("\n" + "="*60)
print("SIG CURVE RECONSTRUCTION DARI SKELETON ZHANG-SUEN")
print("Algoritma: T.Y. Zhang & C.Y. Suen, CACM 27(3), 1984")
print("="*60)

if combined_skeleton is not None and np.any(combined_skeleton > 0):
    num_labels_sig, labels_sig = cv.connectedComponents(
        (combined_skeleton > 0).astype(np.uint8), connectivity=8
    )
    print(f"Jumlah komponen skeleton Zhang-Suen: {num_labels_sig - 1}")

    comp_leftmost = {}
    for comp_id in range(1, num_labels_sig):
        ys_c, xs_c = np.where(labels_sig == comp_id)
        if len(xs_c) > 0:
            comp_leftmost[comp_id] = int(xs_c.min())
    comp_order = sorted(comp_leftmost.keys(), key=lambda c: comp_leftmost[c])

    sig_curves = {}
    comp_display_num = {}

    for display_idx, comp_id in enumerate(comp_order, start=1):
        comp_display_num[comp_id] = display_idx
        mask_comp = ((labels_sig == comp_id).astype(np.uint8) * 255)
        n_pts = int((mask_comp > 0).sum())
        x_left = comp_leftmost[comp_id]
        print(f"\n[Komponen {display_idx} (cv_label={comp_id}, x_kiri={x_left})] "
              f"jumlah titik skeleton Zhang-Suen: {n_pts}")
        if n_pts < 3:
            print(f"  -> Terlalu sedikit titik, skip.")
            continue
        try:
            sig = SIGCurveReconstruction(sigma=1.5)
            curve = sig.fit_from_skeleton_mask(mask_comp)
            sig_curves[display_idx] = (sig, curve, comp_id, x_left)
            print(f"  -> Rekonstruksi berhasil: {len(curve)} titik kurva")
        except Exception as e:
            print(f"  -> Gagal: {e}")
            continue

    # Visualisasi gabungan
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].imshow(combined_skeleton, cmap='gray')
    axes[0].set_title("Hybrid Skeleton Zhang-Suen (Zhang & Suen, 1984)")
    axes[0].axis('off')

    bg_show = gray if len(gray.shape) == 2 else cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    axes[1].imshow(bg_show, cmap='gray', alpha=0.35)
    skel_y, skel_x = np.where(combined_skeleton > 0)
    axes[1].scatter(skel_x, skel_y, c='cyan', s=1, alpha=0.4, label='Zhang-Suen skeleton pixels')

    cmap_comps = plt.cm.tab20(np.linspace(0, 1, max(len(sig_curves), 1)))
    for disp_idx, (sig_obj, curve, cv_id, x_left) in sig_curves.items():
        color = cmap_comps[(disp_idx - 1) % len(cmap_comps)]
        if len(curve) > 1:
            axes[1].plot(curve[:, 0], curve[:, 1], '-', color=color,
                         linewidth=1.8, label=f'Komp-{disp_idx}')
        axes[1].scatter(curve[:, 0], curve[:, 1], c=[color], s=8, zorder=4)

    axes[1].set_title("SIG Curve Reconstruction dari Skeleton Zhang-Suen\n(Zhang & Suen, 1984 | urut kiri→kanan, semua titik masuk)")
    axes[1].axis('off')
    if len(sig_curves) <= 12:
        axes[1].legend(loc='upper right', fontsize=6, markerscale=2, ncol=2)
    plt.tight_layout()
    plt.show()

    # Visualisasi per komponen
    valid_comps = sorted(sig_curves.keys())
    if len(valid_comps) > 0:
        ncols = min(3, len(valid_comps))
        nrows = math.ceil(len(valid_comps) / ncols)
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes2 = np.array(axes2).flatten()
        for plot_idx, disp_idx in enumerate(valid_comps):
            sig_obj, curve, cv_id, x_left = sig_curves[disp_idx]
            n_sk = int(len(sig_obj.points)) if sig_obj.points is not None else 0
            ax = axes2[plot_idx]
            sig_obj.plot_reconstruction(
                ax=ax, show_original=True,
                title=f'Komponen {disp_idx} (x_kiri={x_left})\nZhang-Suen: {n_sk} titik → {len(curve)} pts'
            )
        for idx in range(len(valid_comps), len(axes2)):
            axes2[idx].set_visible(False)
        plt.suptitle("SIG Curve Reconstruction dari Skeleton Zhang-Suen\n"
                     "T.Y. Zhang & C.Y. Suen, Communications of the ACM, 27(3), pp. 236-239, 1984",
                     fontsize=11)
        plt.tight_layout()
        plt.show()

    print("\nSIG Curve Reconstruction (Zhang-Suen) selesai.")
    print(f"Total komponen terekonstruksi: {len(sig_curves)} dari {num_labels_sig - 1}.")
    print("Peta nomor tampilan → cv_label → x_kiri:")
    for disp_idx, (_, _, cv_id, x_left) in sorted(sig_curves.items()):
        print(f"  Komponen {disp_idx:3d} → cv_label={cv_id:3d}, x_kiri={x_left}")

else:
    print("Skeleton kosong, SIG Curve Reconstruction tidak dapat dijalankan.")


# ============================================================
# TAMBAHAN SCRIPT: HOLE CONTOUR
# Digabung dari hole-contour(2).py
# ============================================================

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:15:24 2026

@author: User
"""

import cv2
import numpy as np

# Gunakan image/input yang sudah dibaca oleh script utama
# sehingga tidak perlu membaca file 'Image (30).png' lagi.
hole_source_image = image.copy()
hole_gray = gray.copy() if len(gray.shape) == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(hole_gray, 127, 255, cv2.THRESH_BINARY)

# Find all contours and their hierarchy
# RETR_CCOMP specifically separates external boundaries from holes
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image to draw only holes
hole_img = np.zeros_like(hole_source_image)

if hierarchy is not None:
    # hierarchy[0] contains [Next, Previous, First_Child, Parent]
    for i, h in enumerate(hierarchy[0]):
        # A non-negative parent index (h[3]) means it is an internal hole
        if h[3] != -1:
            cv2.drawContours(hole_img, contours, i, (0, 255, 0), 2)

cv2.imshow('Holes Only', hole_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


if hierarchy is not None:
    # Akses array [0] karena hierarki dibungkus dalam satu list ekstra
    for i, h in enumerate(hierarchy[0]):
        next_c, prev_c, child_c, parent_c = h
        print(f"Kontur #{i}: Next={next_c}, Prev={prev_c}, Child={child_c}, Parent={parent_c}")
        
        
    # Pastikan hierarchy tidak kosong
if hierarchy is not None:
    # Hierarchy berbentuk (1, N, 4), kita ambil [0] untuk akses per kontur
    hierarchy = hierarchy[0] 
    
    for i, contour in enumerate(contours):
        # Ambil info keluarga: [Next, Previous, First_Child, Parent]
        _, _, _, parent_id = hierarchy[i]
        
        # Tentukan Warna:
        # Jika Parent == -1, ini Kontur Luar (Root) -> WARNA BIRU
        # Jika Parent != -1, ini Kontur Dalam (Hole) -> WARNA MERAH
        color = (255, 0, 0) if parent_id == -1 else (0, 0, 255)
        label = f"ID:{i}" if parent_id == -1 else f"ID:{i} (P:{parent_id})"
        
        # 3. Gambar Kontur
        cv2.drawContours(hole_source_image, [contour], -1, color, 2)
        
        # 4. Tambahkan Teks ID di titik pertama kontur
        # Ambil koordinat titik pertama kontur untuk posisi teks
        x, y = contour[0][0]
        # cv2.putText(image, label, (x, y - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# 5. Tampilkan Hasil
cv2.imshow('Hierarchy Visualization (Blue: Root, Red: Child)', hole_source_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
