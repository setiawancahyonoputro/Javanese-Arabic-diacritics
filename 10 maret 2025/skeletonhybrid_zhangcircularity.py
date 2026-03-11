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
import SimpleITK as sitk
from skimage.filters import threshold_local
# medial_axis dihapus — skeletonisasi murni Zhang-Suen (T.Y. Zhang & C.Y. Suen, 1984)
from scipy.ndimage import distance_transform_edt, maximum_filter


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
    Sesuai persepsi dosen:
    Pada lokasi skeleton yang DT > dt_loop_threshold → kembangkan ke
    bentuk threshold awal (binary) pada lokasi tersebut.
    Thinning loop menggunakan Zhang-Suen (1984).
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

    loop_mask   = np.zeros_like(sk, dtype=np.uint8)
    thick_region = np.zeros(fg.shape, dtype=bool)
    region_thick = (dist > dt_loop_threshold) & fg
    thick_region = region_thick

    if region_thick.sum() == 0:
        return sk, loop_mask, thick_region

    n_thick, labels_thick = cv.connectedComponents(
        region_thick.astype(np.uint8), connectivity=8
    )

    for lab_id in range(1, n_thick):
        blob_mask = (labels_thick == lab_id)
        if not np.any(sk_thick & blob_mask):
            continue

        fg_local = fg & blob_mask
        if fg_local.sum() < 5:
            continue

        fg_local_u8 = (fg_local.astype(np.uint8) * 255)
        contours_local, _ = cv.findContours(
            fg_local_u8, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE
        )
        if len(contours_local) == 0:
            continue

        loop_local = np.zeros_like(sk, dtype=np.uint8)
        cv.drawContours(loop_local, contours_local, -1, 1, 1)

        # ← ZHANG: thinning loop menggunakan Zhang-Suen
        loop_thin = zhang_suen_thinning(loop_local > 0).astype(np.uint8)
        loop_mask |= loop_thin

    sk_normal = sk & (~thick_region)
    # ← ZHANG: thinning gabungan dengan Zhang-Suen
    skel_result = zhang_suen_thinning(
        (sk_normal > 0) | (loop_mask > 0)
    ).astype(np.uint8)

    return skel_result, loop_mask, thick_region


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
        # PENTING:
        # fg yang masuk ke sini SUDAH hasil erosi 1px dari blok circular_blob.
        # Jadi permintaan dosen soal erosi tetap terpenuhi.
        #
        # Masalah hasil lama: kontur diambil dari INNER DT region,
        # sehingga loop jatuh terlalu ke dalam (seperti gambar 1).
        # Solusi: setelah erosi, ambil OUTER contour langsung dari fg hasil erosi.
        fg_u8 = (fg.astype(np.uint8) * 255)
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

        # Syarat: skor rata-rata >= 0.5 (artinya mayoritas dimensi "cukup bulat")
        # Ini jauh lebih soft dari if circ > 0.38 and axis > 0.38 and bbox_ratio > 0.35
        if score_round >= 0.50:
            # Pastikan tidak terlalu jauh dari baseline (dy scalable)
            dy_blob_max = dy_max * 2.0  # blob boleh sedikit lebih jauh dari diakritik
            if dy <= dy_blob_max:
                return 'circular_blob', circ, axis, dy

    # 3) NORMAL: semua yang tidak masuk dua kategori di atas
    return 'normal', circ, axis, dy


def preserve_holes(fg255):
    bg = cv.bitwise_not(fg255)
    h, w = bg.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    bg_ff = bg.copy()
    cv.floodFill(bg_ff, mask, (0, 0), 0)
    holes = bg_ff
    return cv.bitwise_and(fg255, cv.bitwise_not(holes))


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

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(image_gray)

T, _ = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

BIAS_HOLE_UP = 35
Th = min(255, int(T) + BIAS_HOLE_UP)
_, th_hole = cv.threshold(img, Th, 255, cv.THRESH_BINARY)
holes0 = get_holes_mask(th_hole)

th_adapt = cv.adaptiveThreshold(
    img, 255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    51,
    -13
)

OTSU_GATE_UP = 4
_, th_gate = cv.threshold(img, min(255, int(T) + OTSU_GATE_UP), 255, cv.THRESH_BINARY)
th_main = cv.bitwise_and(th_adapt, th_gate)

kH = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
th_main = cv.morphologyEx(th_main, cv.MORPH_CLOSE, kH, iterations=1)

kO = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
th_main = cv.morphologyEx(th_main, cv.MORPH_OPEN, kO, iterations=1)

th_main[holes0 == 255] = 0
gray = th_main

gray = split_left_diacritic_pairs(gray, left_frac=0.35, area_max=1200)
gray = clean_noise_keep_text_band(gray, row_frac=0.01, pad_y=40, border=11, min_area_inside=6)
gray = separate_connected_dots_aggressive(gray, max_component_area=250, erosion_size=3)

kernel_dot = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
bw_temp = (gray > 0).astype(np.uint8)
n, labels, stats, _ = cv.connectedComponentsWithStats(bw_temp, connectivity=8)
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 200 and h > w * 1.5:
        component = (labels == i).astype(np.uint8) * 255
        eroded = cv.erode(component, kernel_dot, iterations=1)
        gray[labels == i] = 0
        gray = cv.bitwise_or(gray, eroded)

print("Menghapus noise kecil...")
gray = remove_small_isolated_components(gray, min_area=20, min_width=3, min_height=3)

print("Menghapus dot yang tidak terhubung...")
gray = remove_noise_by_proximity(gray, body_area=1000, body_h=24, dot_area_max=600, dist_keep=22, border=12)

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

_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
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