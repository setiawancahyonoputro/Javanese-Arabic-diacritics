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
from scipy.special import comb
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
# OUTPUT DEBUG BEZIER SEPERTI bezier.py
# ============================================================
# True  -> saat script dijalankan, akan mencetak output numerik
#          len(t) ... bezier xx ... dan xx_sc ... seperti bezier.py.
# False -> matikan jika output terminal terlalu panjang.
PRINT_BEZIER_REFERENCE_DEMO = True
PRINT_BEZIER_PER_HURUF_CALCULATION_OUTPUT = True
BEZIER_SAMPLE_INDICES = (40, 61, 81, 91, 180, 122)

# CSV khusus output perhitungan Bezier.
# Folder dibuat otomatis saat script dijalankan.
SAVE_BEZIER_OUTPUT_CSV = True
CSV_OUTPUT_FOLDER_NAME = "csv"

# ============================================================
# OUTPUT CURVE FITTING
# Semua hasil curve fitting/Bezier per huruf disimpan di folder ini.
# Struktur hasil:
#   curve fitting/
#     *_bezier_perhuruf_grid.png
#     *_bezier_perhuruf_overlay.png      (jika SHOW_BEZIER_OVERLAY=True)
#     per_huruf/
#       huruf_001_bezier.png
#       huruf_002_bezier.png
#       ...
#     csv/
#       *_curve_fitting_summary.csv
#       *_huruf_001_curve_fitting_distance.csv
#       *_huruf_001_bezier_output.csv
#       ...
# ============================================================
CURVE_FITTING_FOLDER_NAME = "curve fitting"
CURVE_FITTING_PER_HURUF_SUBFOLDER = "per_huruf"
CURVE_FITTING_CSV_SUBFOLDER = "csv"


# ============================================================
# TAMBAHAN: PEMOTONGAN HURUF ARAB SEBELUM BEZIER/CURVE FITTING
# ============================================================
# True  -> Bezier dan curve fitting mengambil skeleton dari huruf tunggal
#          yang sudah dipotong memakai definisi neck/seam dekat baseline.
# False -> alur lama tetap dipakai, yaitu dari connected component skeleton gabungan.
USE_ARABIC_LETTER_CUT_FOR_BEZIER = True

# Mode pemotongan:
# - Kandidat utama adalah seam/neck tipis dekat baseline.
# - Penghapusan seam diuji apakah menaikkan jumlah connected component.
# - Jika kandidat strict tidak ditemukan, fallback local-minimum bisa dipakai.
LETTER_CUT_REQUIRE_SPLIT_GAIN = True
LETTER_CUT_ALLOW_LOCAL_MIN_FALLBACK = True

# Titik/diakritik dipisahkan dahulu. Untuk Bezier defaultnya body saja,
# supaya tidak muncul garis kurva semu antara body huruf dan titik.
# Jika ingin skeleton titik ikut dikurvakan, ubah menjadi True.
LETTER_CUT_INCLUDE_MARKS_IN_BEZIER = False

# Debug output segmentasi huruf.
SAVE_ARABIC_LETTER_CUT_DEBUG = True
ARABIC_LETTER_CUT_DEBUG_SUBFOLDER = "arabic_letter_cut"

# Jika deteksi otomatis kurang tepat, isi dengan posisi x global manual.
# Contoh: MANUAL_ARABIC_LETTER_CUT_XS = [120, 146, 188]
MANUAL_ARABIC_LETTER_CUT_XS = []


# ============================================================
# TAMBAHAN SCRIPT: BEZIER CURVE
# Digabung dari bezier.py, disesuaikan agar tidak mengganggu
# alur utama jurnal.py. Fungsi ini bisa dipakai untuk membuat
# koordinat kurva Bezier dan menyimpan/menampilkan plot jika perlu.
# ============================================================
def bezier_basis(i, N, t):
    """Bernstein basis polynomial untuk Bezier curve."""
    t = np.asarray(t, dtype=float)
    return comb(N, i) * (t ** i) * ((1.0 - t) ** (N - i))


def print_bezier_output_like_reference(t, xx, xxs=None, sample_indices=BEZIER_SAMPLE_INDICES):
    """
    Cetak output numerik dengan format yang sama seperti bezier.py:
      len(t) <n> bezier xx <array>
      xx_sc <xxs[40]> <xxs[61]> ...

    xx  = koordinat Bezier sebelum transformasi skala/rotasi/translasi.
    xxs = koordinat setelah transformasi. Jika tidak ada transformasi, isi dengan xx.
    """
    t = np.asarray(t, dtype=float)
    xx = np.asarray(xx, dtype=float)
    if xxs is None:
        xxs = xx
    xxs = np.asarray(xxs, dtype=float)

    print(f"len(t) {len(t)} bezier xx {xx}")

    valid_indices = [idx for idx in sample_indices if 0 <= idx < len(xxs)]
    if valid_indices:
        sample_text = " ".join(str(xxs[idx]) for idx in valid_indices)
        print(f"xx_sc {sample_text}")
    else:
        print("xx_sc")


def _safe_csv_value(value):
    """Buat nilai aman untuk CSV dari numpy/python scalar."""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.ndarray, list, tuple)):
        return str(np.asarray(value, dtype=float).tolist())
    return value


def make_bezier_csv_dir(imagename):
    """Buat folder csv untuk output Bezier."""
    base_dir = os.path.dirname(os.path.abspath(imagename))
    csv_dir = os.path.join(base_dir, CSV_OUTPUT_FOLDER_NAME)
    os.makedirs(csv_dir, exist_ok=True)
    return csv_dir


def _write_rows_csv(path, rows, fieldnames=None):
    """Tulis list of dict ke CSV dengan urutan kolom stabil."""
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_csv_value(row.get(k, "")) for k in fieldnames})


def save_bezier_points_csv(path, t, xx, xxs=None, label="reference", huruf=None):
    """
    Simpan output Bezier dengan struktur seperti bezier.py:
    xx = koordinat Bezier, xx_sc = koordinat setelah transformasi.
    Untuk per-huruf, xx dan xx_sc dibuat sama agar tetap di koordinat huruf/skeleton.
    """
    t = np.asarray(t, dtype=float)
    xx = np.asarray(xx, dtype=float)
    if xxs is None:
        xxs = xx
    xxs = np.asarray(xxs, dtype=float)

    rows = []
    n = min(len(t), len(xx), len(xxs))
    for idx in range(n):
        rows.append({
            "label": label,
            "huruf": "" if huruf is None else huruf,
            "point_index": idx,
            "t": float(t[idx]),
            "bezier_xx_x": float(xx[idx, 0]),
            "bezier_xx_y": float(xx[idx, 1]),
            "xx_sc_x": float(xxs[idx, 0]),
            "xx_sc_y": float(xxs[idx, 1]),
            "is_sample_output_bezier_py": int(idx in BEZIER_SAMPLE_INDICES),
        })

    _write_rows_csv(
        path,
        rows,
        fieldnames=[
            "label", "huruf", "point_index", "t",
            "bezier_xx_x", "bezier_xx_y", "xx_sc_x", "xx_sc_y",
            "is_sample_output_bezier_py",
        ],
    )
    return path


def build_bezier_summary_row(label, t, xx, xxs=None, huruf=None, **metadata):
    """Buat ringkasan, termasuk sample xx_sc seperti output bezier.py."""
    t = np.asarray(t, dtype=float)
    xx = np.asarray(xx, dtype=float)
    if xxs is None:
        xxs = xx
    xxs = np.asarray(xxs, dtype=float)

    row = {
        "label": label,
        "huruf": "" if huruf is None else huruf,
        "len_t": int(len(t)),
        "len_xx": int(len(xx)),
        "len_xx_sc": int(len(xxs)),
    }
    row.update(metadata)

    for idx in BEZIER_SAMPLE_INDICES:
        if 0 <= idx < len(xxs):
            row[f"xx_sc_{idx}_x"] = float(xxs[idx, 0])
            row[f"xx_sc_{idx}_y"] = float(xxs[idx, 1])
        else:
            row[f"xx_sc_{idx}_x"] = ""
            row[f"xx_sc_{idx}_y"] = ""
    return row


def compute_bezier_xx_xxs(t, X, translation=(0, 0), rotation_angle=np.pi / (-360 / 140),
                           scale_xy=(10, 5)):
    """Hitung xx dan xx_sc dengan rumus yang sama seperti bezier.py."""
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X harus berbentuk array/list titik 2D, misal [(x1,y1), (x2,y2), ...]")

    N = X.shape[0] - 1
    xx = np.zeros((len(t), 2), dtype=float)
    for i in range(N + 1):
        xx += np.outer(bezier_basis(i, N, t), X[i])

    r = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)]
    ], dtype=float)
    scale = np.array([[scale_xy[0], 0], [0, scale_xy[1]]], dtype=float)
    xxs = np.matmul(np.matmul(xx, r), scale) + np.asarray(translation, dtype=float)
    return xx, xxs


def bezier_curve(t, X, translation=(0, 0), rotation_angle=np.pi / (-360 / 140),
                 scale_xy=(10, 5), debug=False):
    """
    Evaluasi kurva Bezier dari titik kontrol X.

    Parameters
    ----------
    t : array-like
        Parameter waktu dalam rentang [0, 1].
    X : array-like, shape (n_points, 2)
        Titik kontrol kurva Bezier.
    translation : tuple(float, float)
        Translasi akhir kurva. Bisa diisi centroid/posisi dari jurnal.py.
    rotation_angle : float
        Sudut rotasi dalam radian. Default mengikuti bezier.py.
    scale_xy : tuple(float, float)
        Skala x dan y. Default mengikuti bezier.py.
    debug : bool
        Jika True, cetak ringkasan koordinat.

    Returns
    -------
    np.ndarray
        Koordinat kurva Bezier setelah transformasi, shape (len(t), 2).
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X harus berbentuk array/list titik 2D, misal [(x1,y1), (x2,y2), ...]")

    N = X.shape[0] - 1
    xx = np.zeros((len(t), 2), dtype=float)

    for i in range(N + 1):
        xx += np.outer(bezier_basis(i, N, t), X[i])

    r = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)]
    ], dtype=float)
    scale = np.array([[scale_xy[0], 0], [0, scale_xy[1]]], dtype=float)
    xxs = np.matmul(np.matmul(xx, r), scale) + np.asarray(translation, dtype=float)

    if debug:
        # Format output disamakan dengan bezier.py.
        # Ini mencetak seluruh koordinat Bezier sebelum transformasi (xx)
        # dan beberapa sample koordinat setelah transformasi (xx_sc).
        print_bezier_output_like_reference(t, xx, xxs)

    return xxs


# Alias agar nama fungsi dari bezier.py tetap cocok jika ingin dipakai ulang.
def B(i, N, t):
    return bezier_basis(i, N, t)


def P(t, X, translation):
    # Dibuat debug=True supaya output P(...) sama seperti bezier.py.
    return bezier_curve(t, X, translation=translation, debug=True)


def run_bezier_reference_demo_output(csv_dir=None, image_prefix="bezier_reference"):
    """
    Demo kecil yang sama dengan bezier.py.
    Fungsinya untuk memastikan output hitungan Bezier di bezierr.py
    bisa dibandingkan langsung dengan output referensi yang kamu kirim.
    """
    c = [
        (8.48, 2.88),
        (5.92, 2.73),
        (8.075, 5.23),
        (8.48, 2.88),
    ]
    X = np.array(c)
    translation = (5, 2)
    tt = np.linspace(0, 1, 200)
    xx, xxs = compute_bezier_xx_xxs(tt, X, translation)

    print("\n" + "=" * 60)
    print("OUTPUT HITUNGAN BEZIER REFERENSI - FORMAT bezier.py")
    print("=" * 60)
    print_bezier_output_like_reference(tt, xx, xxs)

    if SAVE_BEZIER_OUTPUT_CSV and csv_dir is not None:
        prefix = os.path.basename(image_prefix) if image_prefix else "bezier_reference"
        reference_csv = os.path.join(csv_dir, f"{prefix}_bezier_reference_output.csv")
        reference_summary_csv = os.path.join(csv_dir, f"{prefix}_bezier_reference_summary.csv")
        save_bezier_points_csv(reference_csv, tt, xx, xxs, label="reference_bezier_py", huruf=None)
        summary_row = build_bezier_summary_row(
            "reference_bezier_py", tt, xx, xxs,
            huruf=None,
            control_points=X.tolist(),
            translation=translation,
            rotation_angle="np.pi/(-360/140)",
            scale_xy=(10, 5),
            csv_file=os.path.basename(reference_csv),
        )
        _write_rows_csv(reference_summary_csv, [summary_row])
        print(f"[BEZIER CSV] Output referensi disimpan: {reference_csv}")
        print(f"[BEZIER CSV] Ringkasan referensi disimpan: {reference_summary_csv}")

    return xxs


def sample_bezier_from_skeleton_points(skeleton_points, num_control=4, num_samples=200,
                                       translation=(0, 0), rotation_angle=0.0,
                                       scale_xy=(1.0, 1.0)):
    """
    Membuat kurva Bezier langsung dari titik skeleton.

    Parameters
    ----------
    skeleton_points : array-like, shape (n_points, 2)
        Titik skeleton dalam format (x, y). Titik diasumsikan sudah terurut
        mengikuti path skeleton.
    num_control : int
        Jumlah titik kontrol yang diambil merata dari skeleton_points.
    num_samples : int
        Jumlah titik hasil sampling kurva Bezier.

    Returns
    -------
    curve : np.ndarray
        Titik kurva Bezier, shape (num_samples, 2).
    control_points : np.ndarray
        Titik kontrol yang dipakai dari skeleton.
    """
    pts = np.asarray(skeleton_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("skeleton_points harus berbentuk array/list titik 2D [(x,y), ...]")
    if len(pts) < 2:
        raise ValueError("Minimal perlu 2 titik skeleton untuk membuat kurva Bezier")

    num_control = int(max(2, min(num_control, len(pts))))
    idx = np.linspace(0, len(pts) - 1, num_control).round().astype(int)
    control_points = pts[idx]
    tt_bezier = np.linspace(0, 1, int(num_samples))
    curve = bezier_curve(
        tt_bezier,
        control_points,
        translation=translation,
        rotation_angle=rotation_angle,
        scale_xy=scale_xy,
        debug=False
    )
    return curve, control_points


# ============================================================
# ============================================================
# BEZIER PER HURUF DARI SKELETON
# Catatan:
# - Tidak memakai demo / titik statis.
# - Kurva dibuat per connected-component skeleton (per huruf/fragmen huruf).
# - Titik skeleton diurutkan lalu diinterpolasi dengan Bezier kubik per segmen
#   agar bentuknya mengikuti skeleton, bukan menjadi satu kurva global.
# ============================================================
def _catmull_rom_to_bezier_path(points, closed=False, samples_per_segment=18):
    """
    Ubah polyline titik skeleton terurut menjadi kurva Bezier kubik piecewise.
    Formula kontrol Catmull-Rom -> Bezier:
      C1 = P1 + (P2 - P0) / 6
      C2 = P2 - (P3 - P1) / 6
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return np.empty((0, 2), dtype=float), []

    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-9:
            keep.append(i)
    pts = pts[keep]
    if len(pts) < 2:
        return np.empty((0, 2), dtype=float), []

    if closed and np.linalg.norm(pts[0] - pts[-1]) <= 1.5:
        pts = pts[:-1]
    if len(pts) < 2:
        return np.empty((0, 2), dtype=float), []

    curve_parts = []
    controls = []
    t = np.linspace(0.0, 1.0, int(samples_per_segment))

    if closed:
        n = len(pts)
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            c1 = p1 + (p2 - p0) / 6.0
            c2 = p2 - (p3 - p1) / 6.0
            controls.append(np.vstack([p1, c1, c2, p2]))
            seg = ((1-t)**3)[:, None] * p1 + (3*((1-t)**2)*t)[:, None] * c1 + \
                  (3*(1-t)*(t**2))[:, None] * c2 + (t**3)[:, None] * p2
            if curve_parts:
                seg = seg[1:]
            curve_parts.append(seg)
        curve = np.vstack(curve_parts)
        curve = np.vstack([curve, curve[0]])
        return curve, controls

    n = len(pts)
    for i in range(n - 1):
        p0 = pts[i - 1] if i - 1 >= 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else pts[i + 1]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        controls.append(np.vstack([p1, c1, c2, p2]))
        seg = ((1-t)**3)[:, None] * p1 + (3*((1-t)**2)*t)[:, None] * c1 + \
              (3*(1-t)*(t**2))[:, None] * c2 + (t**3)[:, None] * p2
        if curve_parts:
            seg = seg[1:]
        curve_parts.append(seg)

    return np.vstack(curve_parts), controls


def _is_skeleton_component_closed(mask_comp):
    """Deteksi sederhana apakah komponen skeleton berupa loop tertutup."""
    sk = (mask_comp > 0).astype(np.uint8)
    ys, xs = np.where(sk > 0)
    if len(xs) < 3:
        return False
    coord_set = set(zip(xs.tolist(), ys.tolist()))
    endpoints = 0
    for x, y in coord_set:
        deg = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in coord_set:
                    deg += 1
        if deg <= 1:
            endpoints += 1
    return endpoints == 0


def make_bezier_curve_per_letter_from_ordered_skeleton(ordered_points, mask_comp=None,
                                                        samples_per_segment=18,
                                                        force_closed=None):
    """Buat kurva Bezier per huruf/komponen dari titik skeleton terurut."""
    pts = np.asarray(ordered_points, dtype=float)
    if len(pts) < 2:
        return np.empty((0, 2), dtype=float), [], False

    if force_closed is None:
        closed = False
        if mask_comp is not None:
            closed = _is_skeleton_component_closed(mask_comp)
        if np.linalg.norm(pts[0] - pts[-1]) <= 2.5:
            closed = True
    else:
        closed = bool(force_closed)

    curve, controls = _catmull_rom_to_bezier_path(
        pts,
        closed=closed,
        samples_per_segment=samples_per_segment
    )
    return curve, controls, closed


# ============================================================
# LOOP-AWARE BEZIER SOURCE DARI SKELETON
# Masalah yang diperbaiki:
# - Skeleton hybrid sudah punya loop dari circular blob detection.
# - Tetapi ordered_points dari traversal DFS bisa membuat loop dibaca sebagai
#   open path, apalagi kalau loop menempel ke stroke lain.
# Solusi:
# - Ambil closed cycle dari graph skeleton sebagai sub-path tersendiri.
# - Closed cycle dipaksa masuk Bezier dengan force_closed=True.
# - Sisa skeleton tanpa cycle tetap diproses sebagai open path biasa.
# ============================================================
BEZIER_EXTRACT_CLOSED_CYCLES = True
BEZIER_MIN_CYCLE_LEN = 8
BEZIER_MIN_CYCLE_AREA = 6.0

# Force loop dari circular_blob detection harus tetap masuk ke Bezier,
# walaupun graph cycle tidak mendeteksi loop karena loop menempel ke stroke
# atau ada gap 1px. Titik tetap berasal dari skeleton hybrid, bukan dari binary.
BEZIER_FORCE_CIRCULAR_BLOB_LOOPS = True
BEZIER_FORCE_LOOP_DILATE_FOR_LABEL = 3
BEZIER_ASSIGN_UNLABELED_SKELETON_TO_NEAREST_LETTER = True


def _ordered_cycle_points_from_nodes(cycle_nodes):
    """Urutkan node cycle menjadi path melingkar stabil untuk Bezier closed."""
    pts = np.asarray(cycle_nodes, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return np.empty((0, 2), dtype=float)

    # Untuk loop circular blob, urutan sudut terhadap centroid paling stabil.
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    pts = pts[np.argsort(ang)]

    # Rotasi agar titik awal konsisten: kiri-atas dulu.
    start_idx = int(np.argmin(pts[:, 0] + 0.001 * pts[:, 1]))
    pts = np.vstack([pts[start_idx:], pts[:start_idx]])
    return pts


def extract_closed_cycle_subpaths_from_skeleton(mask_comp,
                                                min_cycle_len=BEZIER_MIN_CYCLE_LEN,
                                                min_cycle_area=BEZIER_MIN_CYCLE_AREA):
    """
    Ambil loop tertutup dari mask skeleton 1px.

    Return:
      closed_paths : list ndarray Nx2 format (x, y)
      used_pixels  : set((x, y)) piksel loop yang sudah dipakai
    """
    sk = (mask_comp > 0).astype(np.uint8)
    if sk.sum() < 3:
        return [], set()

    G, coords = build_skeleton_pixel_graph(sk)
    if len(coords) < 3 or G.number_of_edges() == 0:
        return [], set()

    cycles = nx.cycle_basis(G)
    closed_paths = []
    used_pixels = set()

    # Cycle yang lebih besar diprioritaskan supaya loop utama tidak kalah oleh
    # cycle kecil akibat 8-neighbor diagonal.
    cycles = sorted(cycles, key=len, reverse=True)

    for cyc in cycles:
        if len(cyc) < int(min_cycle_len):
            continue

        cyc_set = set((int(x), int(y)) for x, y in cyc)
        # Skip cycle yang hampir seluruh pikselnya sudah diambil cycle besar.
        overlap = len(cyc_set & used_pixels) / float(max(1, len(cyc_set)))
        if overlap > 0.70:
            continue

        pts = _ordered_cycle_points_from_nodes(cyc)
        if len(pts) < int(min_cycle_len):
            continue

        contour = pts.astype(np.float32).reshape((-1, 1, 2))
        area = abs(float(cv.contourArea(contour)))
        if area < float(min_cycle_area):
            continue

        closed_paths.append(pts)
        used_pixels.update(cyc_set)

    return closed_paths, used_pixels


def _mask_without_pixels(mask_comp, pixels_to_remove):
    out = (mask_comp > 0).astype(np.uint8)
    H, W = out.shape
    for x, y in pixels_to_remove:
        if 0 <= int(y) < H and 0 <= int(x) < W:
            out[int(y), int(x)] = 0
    return out



def _order_loop_points_from_mask_component(mask_loop_comp):
    """
    Urutkan piksel skeleton loop menjadi path melingkar.

    Dipakai khusus untuk forced loop dari circular blob detection. Loop ini
    sudah dibentuk di skeleton hybrid, jadi Bezier wajib menggambarnya sebagai
    closed sub-path meskipun graph cycle gagal membaca cycle formal.
    """
    sk = (mask_loop_comp > 0).astype(np.uint8)
    ys, xs = np.where(sk > 0)
    if len(xs) < 3:
        return np.empty((0, 2), dtype=float)

    pts = np.column_stack([xs, ys]).astype(float)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    rad = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

    # Sudut adalah kunci agar loop circular blob tetap melingkar di Bezier.
    # Radius hanya tie-breaker untuk piksel dengan sudut hampir sama.
    order = np.lexsort((rad, ang))
    pts = pts[order]

    start_idx = int(np.argmin(pts[:, 0] + 0.001 * pts[:, 1]))
    pts = np.vstack([pts[start_idx:], pts[:start_idx]])
    return pts


def extract_forced_loop_subpaths_from_mask(mask_comp, forced_loop_mask,
                                           min_points=BEZIER_MIN_CYCLE_LEN):
    """
    Ambil loop yang SUDAH ditandai sebagai circular_blob/DT-loop dari skeleton.

    Bedanya dengan cycle_basis:
    - cycle_basis butuh cycle graph yang sempurna;
    - fungsi ini memaksa setiap komponen forced_loop_mask sebagai closed loop,
      karena sumbernya memang berasal dari circular blob detection.
    """
    if forced_loop_mask is None:
        return [], set()

    sk = (mask_comp > 0).astype(np.uint8)
    fm = (forced_loop_mask > 0).astype(np.uint8)
    forced = ((sk > 0) & (fm > 0)).astype(np.uint8)
    if forced.sum() < int(min_points):
        return [], set()

    n, labels = cv.connectedComponents(forced, connectivity=8)
    subpaths = []
    used_pixels = set()

    for lab_id in range(1, n):
        comp = (labels == lab_id).astype(np.uint8)
        if int(comp.sum()) < int(min_points):
            continue
        pts = _order_loop_points_from_mask_component(comp)
        if len(pts) < int(min_points):
            continue
        subpaths.append({
            "kind": "forced_circular_blob_loop",
            "path_index": int(lab_id),
            "points": np.asarray(pts, dtype=float),
            "closed": True,
        })
        ys, xs = np.where(comp > 0)
        used_pixels.update((int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist()))

    return subpaths, used_pixels


def ordered_open_skeleton_points_no_sig(open_mask):
    """
    Urutkan path skeleton terbuka TANPA SIG smoothing.

    Ini menggantikan pemakaian SIGCurveReconstruction untuk Bezier. Titik tetap
    murni dari skeleton; smoothing/fitting dilakukan oleh piecewise Bezier.
    """
    sk = (open_mask > 0).astype(np.uint8)
    G, coords = build_skeleton_pixel_graph(sk)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    ordered = skeleton_graph_to_ordered_path(G, coords)
    if len(ordered) == 0:
        return np.empty((0, 2), dtype=float)
    return np.asarray(ordered, dtype=float)


def build_bezier_subpaths_from_skeleton_mask(mask_comp, ordered_points_fallback=None,
                                             sigma=1.5,
                                             min_open_points=2,
                                             forced_loop_mask=None):
    """
    Pecah satu skeleton huruf menjadi sub-path untuk Bezier:
      - forced_circular_blob_loop -> closed=True
      - cycle graph lain          -> closed=True
      - sisa stroke               -> closed=False

    Penting:
    - Tidak memakai SIGCurveReconstruction.
    - Semua titik berasal dari skeleton mask yang sama.
    - Loop dari circular_blob detection dipaksa tampil sebagai closed sub-path,
      walaupun graph cycle formal gagal.
    """
    sk = (mask_comp > 0).astype(np.uint8)
    subpaths = []
    used_pixels = set()

    # 1) PRIORITAS TERTINGGI: loop yang memang berasal dari circular blob / DT-loop.
    if BEZIER_FORCE_CIRCULAR_BLOB_LOOPS and forced_loop_mask is not None:
        forced_subpaths, forced_used = extract_forced_loop_subpaths_from_mask(
            sk,
            forced_loop_mask,
            min_points=min_open_points,
        )
        subpaths.extend(forced_subpaths)
        used_pixels.update(forced_used)

    # 2) Cycle graph lain pada skeleton yang tersisa.
    sk_for_cycle = _mask_without_pixels(sk, used_pixels)
    if BEZIER_EXTRACT_CLOSED_CYCLES:
        closed_paths, cycle_used = extract_closed_cycle_subpaths_from_skeleton(sk_for_cycle)
        for i, pts in enumerate(closed_paths, start=1):
            subpaths.append({
                "kind": "closed_loop",
                "path_index": i,
                "points": np.asarray(pts, dtype=float),
                "closed": True,
            })
        used_pixels.update(cycle_used)

    # 3) Sisa skeleton diperlakukan sebagai path terbuka, TANPA SIG.
    sk_open = _mask_without_pixels(sk, used_pixels)
    n_open, labels_open = cv.connectedComponents(sk_open.astype(np.uint8), connectivity=8)

    for lab_id in range(1, n_open):
        open_mask = (labels_open == lab_id).astype(np.uint8)
        if int(open_mask.sum()) < int(min_open_points):
            continue

        open_pts = ordered_open_skeleton_points_no_sig(open_mask)
        if open_pts is None or len(open_pts) < int(min_open_points):
            continue

        subpaths.append({
            "kind": "open_stroke",
            "path_index": lab_id,
            "points": np.asarray(open_pts, dtype=float),
            "closed": False,
        })

    # 4) Fallback tanpa SIG jika semua path gagal.
    if not subpaths:
        fallback_pts = None
        if ordered_points_fallback is not None and len(ordered_points_fallback) >= 2:
            fallback_pts = np.asarray(ordered_points_fallback, dtype=float)
        else:
            fallback_pts = ordered_open_skeleton_points_no_sig(sk)

        if fallback_pts is not None and len(fallback_pts) >= 2:
            subpaths.append({
                "kind": "fallback_ordered_no_sig",
                "path_index": 1,
                "points": np.asarray(fallback_pts, dtype=float),
                "closed": _is_skeleton_component_closed(sk),
            })

    return subpaths

# ============================================================
# CURVE FITTING EVALUATION
# Evaluasi ini mengukur seberapa dekat kurva Bezier piecewise
# terhadap titik skeleton huruf. Rumus utama:
#   d(S_j, C) = min_k || S_j - B(t_k) ||_2
#   SSE       = sum_j d(S_j, C)^2
#   RMSE      = sqrt(mean_j d(S_j, C)^2)
# ============================================================
def make_curve_fitting_output_dirs(imagename):
    """Buat folder utama output curve fitting dan subfoldernya."""
    base_dir = os.path.dirname(os.path.abspath(imagename))
    root_dir = os.path.join(base_dir, CURVE_FITTING_FOLDER_NAME)
    per_huruf_dir = os.path.join(root_dir, CURVE_FITTING_PER_HURUF_SUBFOLDER)
    csv_dir = os.path.join(root_dir, CURVE_FITTING_CSV_SUBFOLDER)
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(per_huruf_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    return root_dir, per_huruf_dir, csv_dir


def evaluate_curve_fitting_to_skeleton(skeleton_pts, curve_pts):
    """
    Hitung error pencocokan kurva Bezier terhadap titik skeleton.

    skeleton_pts : titik skeleton terurut, shape (M, 2)
    curve_pts    : titik sampel kurva Bezier, shape (K, 2)

    Output metrik:
    - SSE/RMSE/MAE dari skeleton ke kurva
    - SSE/RMSE/MAE dari kurva ke skeleton
    - symmetric Hausdorff distance
    """
    skeleton_pts = np.asarray(skeleton_pts, dtype=float)
    curve_pts = np.asarray(curve_pts, dtype=float)

    if skeleton_pts.ndim != 2 or curve_pts.ndim != 2:
        raise ValueError("skeleton_pts dan curve_pts harus array 2D")
    if skeleton_pts.shape[1] != 2 or curve_pts.shape[1] != 2:
        raise ValueError("skeleton_pts dan curve_pts harus berisi koordinat (x, y)")
    if len(skeleton_pts) == 0 or len(curve_pts) == 0:
        return {
            "sse_skeleton_to_curve": 0.0,
            "rmse_skeleton_to_curve": 0.0,
            "mae_skeleton_to_curve": 0.0,
            "max_skeleton_to_curve": 0.0,
            "sse_curve_to_skeleton": 0.0,
            "rmse_curve_to_skeleton": 0.0,
            "mae_curve_to_skeleton": 0.0,
            "max_curve_to_skeleton": 0.0,
            "hausdorff_symmetric": 0.0,
        }

    tree_curve = cKDTree(curve_pts)
    dist_skel_to_curve, _ = tree_curve.query(skeleton_pts)

    tree_skeleton = cKDTree(skeleton_pts)
    dist_curve_to_skel, _ = tree_skeleton.query(curve_pts)

    sse_sc = float(np.sum(dist_skel_to_curve ** 2))
    rmse_sc = float(np.sqrt(np.mean(dist_skel_to_curve ** 2)))
    mae_sc = float(np.mean(dist_skel_to_curve))
    max_sc = float(np.max(dist_skel_to_curve))

    sse_cs = float(np.sum(dist_curve_to_skel ** 2))
    rmse_cs = float(np.sqrt(np.mean(dist_curve_to_skel ** 2)))
    mae_cs = float(np.mean(dist_curve_to_skel))
    max_cs = float(np.max(dist_curve_to_skel))

    return {
        "sse_skeleton_to_curve": sse_sc,
        "rmse_skeleton_to_curve": rmse_sc,
        "mae_skeleton_to_curve": mae_sc,
        "max_skeleton_to_curve": max_sc,
        "sse_curve_to_skeleton": sse_cs,
        "rmse_curve_to_skeleton": rmse_cs,
        "mae_curve_to_skeleton": mae_cs,
        "max_curve_to_skeleton": max_cs,
        "hausdorff_symmetric": float(max(max_sc, max_cs)),
    }


def save_curve_fitting_distance_csv(path, skeleton_pts, curve_pts, huruf=None):
    """
    Simpan jarak tiap titik skeleton ke titik kurva Bezier terdekat.
    File ini berguna untuk bukti numerik bahwa kurva dicocokkan ke skeleton.
    """
    skeleton_pts = np.asarray(skeleton_pts, dtype=float)
    curve_pts = np.asarray(curve_pts, dtype=float)

    rows = []
    if len(skeleton_pts) > 0 and len(curve_pts) > 0:
        tree_curve = cKDTree(curve_pts)
        distances, nearest_idx = tree_curve.query(skeleton_pts)
        for i, (pt, d, j) in enumerate(zip(skeleton_pts, distances, nearest_idx)):
            nearest = curve_pts[int(j)]
            rows.append({
                "huruf": "" if huruf is None else int(huruf),
                "skeleton_index": int(i),
                "skeleton_x": float(pt[0]),
                "skeleton_y": float(pt[1]),
                "nearest_curve_index": int(j),
                "nearest_curve_x": float(nearest[0]),
                "nearest_curve_y": float(nearest[1]),
                "distance_px": float(d),
                "distance_squared_px": float(d * d),
            })

    _write_rows_csv(
        path,
        rows,
        fieldnames=[
            "huruf", "skeleton_index", "skeleton_x", "skeleton_y",
            "nearest_curve_index", "nearest_curve_x", "nearest_curve_y",
            "distance_px", "distance_squared_px",
        ],
    )
    return path


def build_curve_fitting_summary_row(huruf, skeleton_pts, curve_pts, control_segments,
                                    closed_flag, cv_id, x_left, fit_eval,
                                    distance_csv_file=""):
    """Buat satu baris ringkasan curve fitting untuk satu huruf/komponen."""
    row = {
        "huruf": int(huruf),
        "cv_label": int(cv_id),
        "x_left": float(x_left),
        "skeleton_points": int(len(skeleton_pts)),
        "curve_points": int(len(curve_pts)),
        "segment_count": int(len(control_segments)),
        "closed": bool(closed_flag),
        "distance_csv_file": distance_csv_file,
    }
    row.update({k: float(v) for k, v in fit_eval.items()})
    return row


# ============================================================
# TAMBAHAN: ARABIC LETTER CUTTING UNTUK SUMBER BEZIER
# Definisi yang dipakai:
# cut point = seam/neck tipis dekat baseline, berada di antara dua massa huruf,
# dan jika seam kecil dihapus jumlah connected component bertambah.
# Output fungsi ini adalah skeleton huruf tunggal yang sudah dipotong.
# ============================================================
def _arabic_cut_output_dir(imagename):
    base_dir = os.path.dirname(os.path.abspath(imagename))
    out_dir = os.path.join(base_dir, ARABIC_LETTER_CUT_DEBUG_SUBFOLDER)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _component_bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _bbox_distance_to_centroid(mask_bool, cx, cy):
    x, y, w, h = _component_bbox_from_mask(mask_bool)
    if w <= 0 or h <= 0:
        return float('inf')
    dx = 0.0
    if cx < x:
        dx = float(x - cx)
    elif cx > x + w - 1:
        dx = float(cx - (x + w - 1))
    dy = 0.0
    if cy < y:
        dy = float(y - cy)
    elif cy > y + h - 1:
        dy = float(cy - (y + h - 1))
    return math.sqrt(dx * dx + dy * dy)


def split_body_and_marks_for_arabic_cut(binary_img, baseline_y, median_body_h, r_stroke=1.5):
    """
    Pisahkan body/rasm dan titik/diakritik sebelum pemotongan huruf.

    Body dipakai untuk mencari cut seam. Mark/dot disimpan terpisah agar tidak
    mengganggu vertical projection dan connected component splitting.
    """
    bw = (binary_img > 0).astype(np.uint8)
    H2, W2 = bw.shape
    body = np.zeros_like(bw, dtype=np.uint8)
    marks = np.zeros_like(bw, dtype=np.uint8)

    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return body, marks

    r = max(1.0, float(r_stroke))
    med_h = max(1.0, float(median_body_h))
    body_area_min = max(28, int(round(7.0 * r * r)))
    body_h_min = max(8, int(round(4.5 * r)))
    body_w_min = max(10, int(round(0.018 * W2)))
    mark_area_max = max(60, int(round(75.0 * r * r)))
    baseline_band = max(8.0, 0.45 * med_h, 3.2 * r)

    # First pass: classify each binary component.
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = cents[i]
        dy = abs(float(cy) - float(baseline_y))
        aspect = float(max(w, h) / (min(w, h) + 1e-9))
        near_baseline = dy <= baseline_band
        large_or_stroke = (area >= body_area_min) or (h >= body_h_min) or (w >= body_w_min)
        dot_like_far = (
            (area <= mark_area_max) and
            (not near_baseline) and
            (aspect < 5.0) and
            (h <= 0.95 * med_h) and
            (w <= 0.95 * med_h)
        )

        if dot_like_far:
            marks[labels == i] = 1
        else:
            body[labels == i] = 1

    # Fallback: if body is empty, keep the largest component as body.
    if body.sum() == 0 and n > 1:
        largest_id = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        body[labels == largest_id] = 1
        marks[(bw > 0) & (labels != largest_id)] = 1

    return (body > 0).astype(np.uint8), (marks > 0).astype(np.uint8)


def _removal_split_gain_for_cut(crop_bool, x_local, baseline_local_y, seam_half_width,
                                baseline_band, min_area):
    """Uji definisi cut: seam kecil dihapus, CC bertambah dan sisi kiri-kanan valid."""
    crop = (crop_bool > 0).astype(np.uint8)
    if crop.sum() == 0:
        return 0, 0, False

    h, w = crop.shape
    x0 = max(0, int(x_local) - int(seam_half_width))
    x1 = min(w - 1, int(x_local) + int(seam_half_width))
    y0 = max(0, int(round(baseline_local_y - baseline_band)))
    y1 = min(h - 1, int(round(baseline_local_y + baseline_band)))

    before_n, _, before_stats, _ = cv.connectedComponentsWithStats(crop, connectivity=8)
    before_valid = sum(1 for i in range(1, before_n) if before_stats[i, cv.CC_STAT_AREA] >= min_area)

    test = crop.copy()
    test[y0:y1 + 1, x0:x1 + 1] = 0
    after_n, after_labels, after_stats, after_cents = cv.connectedComponentsWithStats(test, connectivity=8)

    valid_ids = []
    left_ok = False
    right_ok = False
    for i in range(1, after_n):
        area = int(after_stats[i, cv.CC_STAT_AREA])
        if area < min_area:
            continue
        valid_ids.append(i)
        cx = float(after_cents[i][0])
        if cx < x_local - seam_half_width:
            left_ok = True
        if cx > x_local + seam_half_width:
            right_ok = True

    split_gain = max(0, len(valid_ids) - max(1, before_valid))
    split_ok = (len(valid_ids) >= 2) and left_ok and right_ok
    return int(split_gain), int(len(valid_ids)), bool(split_ok)


def _score_arabic_cut_candidate(crop_bool, skel_bool, x_local, baseline_local_y,
                                r_stroke, min_area):
    crop = (crop_bool > 0).astype(np.uint8)
    skel = (skel_bool > 0).astype(np.uint8)
    h, w = crop.shape
    r = max(1.0, float(r_stroke))
    band = max(3, int(round(3.0 * r)))
    y0 = max(0, int(round(baseline_local_y - band)))
    y1 = min(h - 1, int(round(baseline_local_y + band)))
    seam_half = max(1, int(round(0.7 * r)))

    col_ink = crop.sum(axis=0).astype(float)
    band_ink = crop[y0:y1 + 1, :].sum(axis=0).astype(float)
    skel_band = skel[y0:y1 + 1, :].sum(axis=0).astype(float)

    max_col = max(1.0, float(np.percentile(col_ink[col_ink > 0], 90)) if np.any(col_ink > 0) else 1.0)
    max_band = max(1.0, float(np.percentile(band_ink[band_ink > 0], 90)) if np.any(band_ink > 0) else 1.0)
    max_skel = max(1.0, float(np.percentile(skel_band[skel_band > 0], 90)) if np.any(skel_band > 0) else 1.0)

    if crop[:, x_local].sum() > 0:
        ys = np.where(crop[:, x_local] > 0)[0]
        dist_base = abs(float(np.median(ys)) - float(baseline_local_y))
    else:
        dist_base = band

    split_gain, valid_after, split_ok = _removal_split_gain_for_cut(
        crop,
        x_local,
        baseline_local_y,
        seam_half,
        band,
        min_area=min_area,
    )

    score = (
        0.45 * (col_ink[x_local] / max_col) +
        0.35 * (band_ink[x_local] / max_band) +
        0.20 * (skel_band[x_local] / max_skel) +
        0.10 * (dist_base / max(1.0, float(band))) -
        0.70 * float(split_ok) -
        0.20 * float(split_gain)
    )
    return {
        "x_local": int(x_local),
        "score": float(score),
        "col_ink": float(col_ink[x_local]),
        "band_ink": float(band_ink[x_local]),
        "skel_band_ink": float(skel_band[x_local]),
        "split_gain": int(split_gain),
        "valid_components_after_removal": int(valid_after),
        "split_ok": bool(split_ok),
        "distance_from_baseline": float(dist_base),
    }


def find_cut_points_in_body_component(component_bool, baseline_y_global, x_offset, y_offset,
                                      r_stroke=1.5, require_split=True,
                                      allow_fallback=True):
    """Cari cut x global pada satu connected component body/rasm."""
    comp = (component_bool > 0).astype(np.uint8)
    ys_all, xs_all = np.where(comp > 0)
    if len(xs_all) == 0:
        return [], []

    x, y, w, h = _component_bbox_from_mask(comp)
    crop = comp[y:y + h, x:x + w]
    baseline_local_y = float(baseline_y_global - (y_offset + y))
    r = max(1.0, float(r_stroke))
    min_interval_w = max(14, int(round(5.0 * r)))
    min_area = max(10, int(round(5.0 * r * r)))

    if w < (2 * min_interval_w + 2) or int(crop.sum()) < (2 * min_area):
        return [], []

    skel_crop = zhang_suen_thinning(crop > 0).astype(np.uint8)
    col_ink = crop.sum(axis=0).astype(float)
    band = max(3, int(round(3.0 * r)))
    y0 = max(0, int(round(baseline_local_y - band)))
    y1 = min(h - 1, int(round(baseline_local_y + band)))
    band_ink = crop[y0:y1 + 1, :].sum(axis=0).astype(float)

    win = max(2, int(round(2.5 * r)))
    margin = min_interval_w
    candidate_rows = []

    for xx in range(margin, w - margin):
        local0 = max(0, xx - win)
        local1 = min(w, xx + win + 1)
        local_col = col_ink[local0:local1]
        local_band = band_ink[local0:local1]
        is_local_min = (col_ink[xx] <= np.min(local_col) + 1.0) or (band_ink[xx] <= np.min(local_band) + 1.0)
        if not is_local_min:
            continue
        if col_ink[xx] <= 0 and band_ink[xx] <= 0:
            continue

        # Definisi "di antara dua badan huruf": sisi kiri dan kanan harus
        # punya massa lebih tebal daripada seam. Ini mencegah over-cut pada
        # stroke panjang halus yang hanya berupa garis baseline.
        context = max(min_interval_w, int(round(8.0 * r)))
        gap = max(2, int(round(1.5 * r)))
        l0 = max(0, xx - context)
        l1 = max(0, xx - gap)
        r0 = min(w, xx + gap + 1)
        r1 = min(w, xx + context + 1)
        left_band_peak = float(np.max(band_ink[l0:l1])) if l1 > l0 else 0.0
        right_band_peak = float(np.max(band_ink[r0:r1])) if r1 > r0 else 0.0
        left_col_peak = float(np.max(col_ink[l0:l1])) if l1 > l0 else 0.0
        right_col_peak = float(np.max(col_ink[r0:r1])) if r1 > r0 else 0.0
        seam_band = float(band_ink[xx])
        seam_col = float(col_ink[xx])
        band_contrast = min(left_band_peak, right_band_peak) - seam_band
        col_contrast = min(left_col_peak, right_col_peak) - seam_col
        band_ratio = min(left_band_peak, right_band_peak) / (seam_band + 1.0)
        col_ratio = min(left_col_peak, right_col_peak) / (seam_col + 1.0)
        mass_ok = (
            (band_contrast >= max(4.0, 1.8 * r) and band_ratio >= 1.65) or
            (col_contrast >= max(6.0, 2.5 * r) and col_ratio >= 1.45)
        )

        row = _score_arabic_cut_candidate(crop, skel_crop, xx, baseline_local_y, r, min_area=min_area)
        row["left_band_peak"] = float(left_band_peak)
        row["right_band_peak"] = float(right_band_peak)
        row["left_col_peak"] = float(left_col_peak)
        row["right_col_peak"] = float(right_col_peak)
        row["band_contrast"] = float(band_contrast)
        row["col_contrast"] = float(col_contrast)
        row["band_ratio"] = float(band_ratio)
        row["col_ratio"] = float(col_ratio)
        row["mass_ok"] = int(bool(mass_ok))

        # Candidate harus split_ok dan berada di antara dua massa. Fallback
        # local-minimum hanya dipakai jika tidak ada strict candidate.
        if row["split_ok"] and mass_ok:
            candidate_rows.append(row)
        elif allow_fallback and mass_ok and (row["score"] <= 0.85):
            candidate_rows.append(row)

    if not candidate_rows:
        return [], []

    strict_rows = [r0 for r0 in candidate_rows if r0["split_ok"] and bool(r0.get("mass_ok", 0))]
    if require_split and strict_rows:
        usable_rows = strict_rows
    elif require_split and (not strict_rows) and allow_fallback:
        usable_rows = [r0 for r0 in candidate_rows if bool(r0.get("mass_ok", 0))]
    elif require_split:
        usable_rows = []
    else:
        usable_rows = [r0 for r0 in candidate_rows if bool(r0.get("mass_ok", 0))]

    # Lower score is better. Keep non-overlapping cuts that leave valid intervals.
    usable_rows = sorted(usable_rows, key=lambda rr: (rr["score"], rr["x_local"]))
    selected = []

    def _intervals_are_valid(cuts_local):
        bounds = [0] + sorted(cuts_local) + [w]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if (b - a) < min_interval_w:
                return False
            if int(crop[:, a:b].sum()) < min_area:
                return False
        return True

    max_cuts = max(0, int(w / float(min_interval_w)) - 1)
    for row in usable_rows:
        xx = int(row["x_local"])
        if any(abs(xx - s) < min_interval_w for s in selected):
            continue
        proposed = sorted(selected + [xx])
        if not _intervals_are_valid(proposed):
            continue
        selected.append(xx)
        if len(selected) >= max_cuts:
            break

    selected = sorted(selected)
    selected_global = [int(x_offset + x + xx) for xx in selected]

    # Return metadata in global coordinates.
    meta = []
    for row in candidate_rows:
        row2 = dict(row)
        row2["x_global"] = int(x_offset + x + int(row["x_local"]))
        row2["component_bbox_x"] = int(x_offset + x)
        row2["component_bbox_y"] = int(y_offset + y)
        row2["component_bbox_w"] = int(w)
        row2["component_bbox_h"] = int(h)
        row2["selected"] = int(row2["x_global"] in selected_global)
        meta.append(row2)

    return selected_global, meta


def build_cut_letter_masks_from_body(body_mask, baseline_y, median_body_h, r_stroke=1.5,
                                     manual_cut_xs=None):
    """Potong body/rasm menjadi mask huruf tunggal berdasarkan cut points."""
    body = (body_mask > 0).astype(np.uint8)
    H2, W2 = body.shape
    n, labels, stats, _ = cv.connectedComponentsWithStats(body, connectivity=8)

    letter_masks = []
    cut_rows = []
    manual_cut_xs = sorted([int(x) for x in (manual_cut_xs or [])])

    for comp_id in range(1, n):
        comp = (labels == comp_id).astype(np.uint8)
        if int(comp.sum()) <= 0:
            continue
        x, y, w, h = _component_bbox_from_mask(comp)

        auto_cuts, meta_rows = find_cut_points_in_body_component(
            component_bool=comp,
            baseline_y_global=baseline_y,
            x_offset=0,
            y_offset=0,
            r_stroke=r_stroke,
            require_split=LETTER_CUT_REQUIRE_SPLIT_GAIN,
            allow_fallback=LETTER_CUT_ALLOW_LOCAL_MIN_FALLBACK,
        )
        cut_rows.extend(meta_rows)

        # Add manual cuts that fall inside this component bbox.
        comp_manual = [cx for cx in manual_cut_xs if x < cx < (x + w - 1)]
        cuts = sorted(set(auto_cuts + comp_manual))

        bounds = [x] + cuts + [x + w]
        for a, b in zip(bounds[:-1], bounds[1:]):
            sub = np.zeros_like(body, dtype=np.uint8)
            # Remove a 1px seam on each selected cut by using half-open intervals.
            sub[:, a:b] = comp[:, a:b]
            if int(sub.sum()) <= 0:
                continue
            sx, sy, sw, sh = _component_bbox_from_mask(sub)
            min_area = max(8, int(round(4.0 * max(1.0, r_stroke) ** 2)))
            if int(sub.sum()) < min_area:
                continue
            letter_masks.append({
                "body_mask": (sub > 0),
                "full_mask": (sub > 0),
                "bbox": (int(sx), int(sy), int(sw), int(sh)),
                "source_component": int(comp_id),
                "cut_left": int(a),
                "cut_right": int(b),
            })

    # Sort Arabic text right-to-left for logical processing, but keep display index stable.
    # The plotting code later still sorts by x_left. Here we preserve masks left-to-right
    # because existing script names Huruf 001 from smaller x.
    letter_masks.sort(key=lambda m: (m["bbox"][0], m["bbox"][1]))
    return letter_masks, cut_rows


def assign_marks_to_cut_letters(letter_masks, marks_mask):
    """Assign titik/diakritik ke huruf terdekat setelah body dipotong."""
    marks = (marks_mask > 0).astype(np.uint8)
    if len(letter_masks) == 0 or marks.sum() == 0:
        return letter_masks, []

    n, labels, stats, cents = cv.connectedComponentsWithStats(marks, connectivity=8)
    assignment_rows = []

    for mark_id in range(1, n):
        comp = (labels == mark_id)
        if not np.any(comp):
            continue
        cx, cy = cents[mark_id]
        best_idx = None
        best_dist = float('inf')
        for idx_l, item in enumerate(letter_masks):
            body = item["body_mask"]
            dist = _bbox_distance_to_centroid(body, cx, cy)
            bx, by, bw, bh = item["bbox"]
            # Horizontal overlap gets a bonus because Arabic dots usually belong
            # to the nearest body under/above the same x-range.
            if bx - 4 <= cx <= bx + bw + 4:
                dist *= 0.55
            if dist < best_dist:
                best_dist = dist
                best_idx = idx_l
        if best_idx is not None:
            letter_masks[best_idx]["full_mask"] = (letter_masks[best_idx]["full_mask"] | comp)
            assignment_rows.append({
                "mark_id": int(mark_id),
                "assigned_letter_index": int(best_idx + 1),
                "mark_area": int(stats[mark_id, cv.CC_STAT_AREA]),
                "mark_cx": float(cx),
                "mark_cy": float(cy),
                "distance_score": float(best_dist),
            })

    return letter_masks, assignment_rows


def make_bezier_skeleton_source_from_cut_letters(binary_img, baseline_y, median_body_h,
                                                 r_stroke_global, imagename,
                                                 include_marks_in_bezier=False,
                                                 debug=True,
                                                 base_skeleton_for_bezier=None,
                                                 forced_loop_skeleton=None):
    """
    Bangun skeleton + label source untuk SIG/Bezier dari huruf tunggal.

    Return:
      source_skeleton : uint8 0/255, skeleton huruf hasil potong
      source_labels   : int32 label per huruf, dapat melabeli subpath terpisah
      info            : dict debug dan ringkasan
    """
    bw = (binary_img > 0).astype(np.uint8)
    H2, W2 = bw.shape
    body_mask, marks_mask = split_body_and_marks_for_arabic_cut(
        bw, baseline_y=baseline_y, median_body_h=median_body_h,
        r_stroke=r_stroke_global
    )

    letter_masks, cut_rows = build_cut_letter_masks_from_body(
        body_mask,
        baseline_y=baseline_y,
        median_body_h=median_body_h,
        r_stroke=r_stroke_global,
        manual_cut_xs=MANUAL_ARABIC_LETTER_CUT_XS,
    )
    letter_masks, mark_rows = assign_marks_to_cut_letters(letter_masks, marks_mask)

    source_skeleton = np.zeros((H2, W2), dtype=np.uint8)
    source_labels = np.zeros((H2, W2), dtype=np.int32)
    summary_rows = []

    # KUNCI PERBAIKAN:
    # Jika base_skeleton_for_bezier diberikan, source Bezier diambil dari
    # skeleton hybrid final, bukan skeleton ulang dari binary cut. Inilah yang
    # menjaga loop circular_blob tetap muncul pada kurva Bezier.
    if base_skeleton_for_bezier is not None and np.any(np.asarray(base_skeleton_for_bezier) > 0):
        base_skel = (np.asarray(base_skeleton_for_bezier) > 0).astype(np.uint8)
        use_base_skeleton = True
    else:
        base_skel = None
        use_base_skeleton = False

    forced_loop = np.zeros((H2, W2), dtype=np.uint8)
    if forced_loop_skeleton is not None:
        forced_loop = (np.asarray(forced_loop_skeleton) > 0).astype(np.uint8)

    k_label = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (max(1, int(BEZIER_FORCE_LOOP_DILATE_FOR_LABEL)), max(1, int(BEZIER_FORCE_LOOP_DILATE_FOR_LABEL)))
    )

    for idx_l, item in enumerate(letter_masks, start=1):
        src_mask = item["full_mask"] if include_marks_in_bezier else item["body_mask"]
        src_mask = (src_mask > 0)
        if not np.any(src_mask):
            continue

        if use_base_skeleton:
            # Ambil skeleton hybrid yang berada di area huruf hasil cut.
            label_region = cv.dilate(src_mask.astype(np.uint8), k_label, iterations=1) > 0
            sk = ((base_skel > 0) & label_region).astype(np.uint8)

            # Jika forced loop bersinggungan dengan bbox huruf, ikutkan juga.
            x0, y0, w0, h0 = item["bbox"]
            bbox_region = np.zeros_like(src_mask, dtype=np.uint8)
            pad = max(3, int(round(3.0 * max(1.0, float(r_stroke_global)))))
            bx0 = max(0, int(x0) - pad)
            by0 = max(0, int(y0) - pad)
            bx1 = min(W2, int(x0 + w0) + pad)
            by1 = min(H2, int(y0 + h0) + pad)
            bbox_region[by0:by1, bx0:bx1] = 1
            sk |= ((forced_loop > 0) & (bbox_region > 0)).astype(np.uint8)

            # Fallback hanya kalau base skeleton benar-benar tidak memberi piksel.
            if not np.any(sk > 0):
                sk = zhang_suen_thinning(src_mask).astype(np.uint8)
        else:
            sk = zhang_suen_thinning(src_mask).astype(np.uint8)

        sk = (sk > 0).astype(np.uint8)
        source_skeleton[sk > 0] = 255
        source_labels[sk > 0] = int(idx_l)

    # Assign skeleton hybrid yang belum berlabel ke huruf terdekat.
    # Ini wajib untuk circular blob loop yang berada sebagai komponen terpisah
    # atau jatuh sedikit di luar body_mask cut-letter.
    if use_base_skeleton and BEZIER_ASSIGN_UNLABELED_SKELETON_TO_NEAREST_LETTER and len(letter_masks) > 0:
        unassigned = ((base_skel > 0) & (source_labels == 0)).astype(np.uint8)
        if np.any(unassigned > 0):
            n_un, lab_un, st_un, cent_un = cv.connectedComponentsWithStats(unassigned, connectivity=8)
            for uid in range(1, n_un):
                comp = (lab_un == uid)
                if not np.any(comp):
                    continue
                area_u = int(st_un[uid, cv.CC_STAT_AREA])
                cx_u, cy_u = cent_un[uid]
                comp_forced = np.count_nonzero((forced_loop > 0) & comp) > 0

                best_idx = None
                best_dist = float('inf')
                for idx0, item in enumerate(letter_masks, start=1):
                    ref_mask = item["full_mask"] if include_marks_in_bezier else item["body_mask"]
                    d = _bbox_distance_to_centroid(ref_mask, cx_u, cy_u)
                    bx, by, bw0, bh0 = item["bbox"]
                    if bx - 4 <= cx_u <= bx + bw0 + 4:
                        d *= 0.55
                    if d < best_dist:
                        best_dist = d
                        best_idx = idx0

                # Forced circular loop selalu di-assign. Non-forced hanya jika dekat,
                # supaya titik/diakritik jauh tidak ikut ketika include_marks=False.
                near_limit = max(6.0, 4.0 * max(1.0, float(r_stroke_global)))
                if best_idx is not None and (comp_forced or include_marks_in_bezier or best_dist <= near_limit):
                    source_skeleton[comp] = 255
                    source_labels[comp] = int(best_idx)

    # Buat summary setelah seluruh assignment selesai.
    for idx_l, item in enumerate(letter_masks, start=1):
        x, y, w, h = item["bbox"]
        sk_count = int(np.count_nonzero(source_labels == idx_l))
        summary_rows.append({
            "huruf": int(idx_l),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "source_component": int(item.get("source_component", -1)),
            "cut_left": int(item.get("cut_left", x)),
            "cut_right": int(item.get("cut_right", x + w)),
            "body_area_px": int(np.count_nonzero(item["body_mask"])),
            "full_area_px": int(np.count_nonzero(item["full_mask"])),
            "skeleton_points": int(sk_count),
            "marks_included_in_bezier": int(bool(include_marks_in_bezier)),
            "skeleton_source": "hybrid_base_skeleton" if use_base_skeleton else "zhang_suen_from_cut_binary",
            "forced_loop_points": int(np.count_nonzero((source_labels == idx_l) & (forced_loop > 0))),
        })

    info = {
        "body_mask": body_mask,
        "marks_mask": marks_mask,
        "letter_masks": letter_masks,
        "cut_rows": cut_rows,
        "mark_rows": mark_rows,
        "summary_rows": summary_rows,
    }

    if debug and SAVE_ARABIC_LETTER_CUT_DEBUG:
        try:
            out_dir = _arabic_cut_output_dir(imagename)
            prefix = os.path.basename(imagename)

            # Save debug images.
            cv.imwrite(os.path.join(out_dir, f"{prefix}_body_mask.png"), (body_mask > 0).astype(np.uint8) * 255)
            cv.imwrite(os.path.join(out_dir, f"{prefix}_marks_mask.png"), (marks_mask > 0).astype(np.uint8) * 255)
            cv.imwrite(os.path.join(out_dir, f"{prefix}_bezier_source_skeleton.png"), source_skeleton)

            # Label visualization.
            label_vis = np.zeros((H2, W2, 3), dtype=np.uint8)
            rng_colors = []
            for k in range(max(1, len(letter_masks))):
                rng_colors.append(((37 * (k + 3)) % 255, (83 * (k + 5)) % 255, (151 * (k + 7)) % 255))
            for idx_l in range(1, int(source_labels.max()) + 1):
                color = rng_colors[(idx_l - 1) % len(rng_colors)]
                label_vis[source_labels == idx_l] = color
            cv.imwrite(os.path.join(out_dir, f"{prefix}_bezier_source_labels.png"), cv.cvtColor(label_vis, cv.COLOR_RGB2BGR))

            # Overlay cuts on binary image.
            canvas = cv.cvtColor((bw * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
            selected_xs = sorted(set(int(r.get("x_global", -1)) for r in cut_rows if int(r.get("selected", 0)) == 1))
            for cx in selected_xs + [int(x) for x in MANUAL_ARABIC_LETTER_CUT_XS]:
                if 0 <= cx < W2:
                    cv.line(canvas, (cx, 0), (cx, H2 - 1), (0, 0, 255), 1)
            for idx_l, item in enumerate(letter_masks, start=1):
                x, y, w, h = item["bbox"]
                cv.rectangle(canvas, (x, y), (x + w, y + h), (0, 180, 0), 1)
                cv.putText(canvas, str(idx_l), (x, max(10, y - 2)), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0), 1, cv.LINE_AA)
            cv.imwrite(os.path.join(out_dir, f"{prefix}_cut_overlay.png"), canvas)

            # CSV summaries.
            if summary_rows:
                _write_rows_csv(os.path.join(out_dir, f"{prefix}_letter_cut_summary.csv"), summary_rows)
            if cut_rows:
                _write_rows_csv(os.path.join(out_dir, f"{prefix}_cut_candidates.csv"), cut_rows)
            if mark_rows:
                _write_rows_csv(os.path.join(out_dir, f"{prefix}_mark_assignment.csv"), mark_rows)

            print(f"[ARABIC LETTER CUT] Debug output disimpan di: {out_dir}")
        except Exception as e:
            print(f"[ARABIC LETTER CUT] Gagal menyimpan debug output: {e}")

    print(
        f"[ARABIC LETTER CUT] huruf={len(summary_rows)} | "
        f"cut_selected={sum(int(r.get('selected', 0)) for r in cut_rows)} | "
        f"skeleton_points={int(np.count_nonzero(source_skeleton))} | "
        f"include_marks_in_bezier={bool(include_marks_in_bezier)}"
    )

    return source_skeleton, source_labels, info

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

# Warna skeleton final untuk visualisasi Matplotlib/RGB:
# - rasm / badan huruf        = hijau
# - diakritik / titik harakat = merah
SKELETON_RASM_RGB = np.array([0, 255, 0], dtype=np.uint8)
SKELETON_DIACRITIC_RGB = np.array([255, 0, 0], dtype=np.uint8)

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
        "diacritic": (0, 0, 255),       # merah untuk diakritik (BGR/OpenCV)
        "circular_blob": (0, 180, 0),   # hijau untuk rasm/blob
        "normal": (0, 180, 0),          # hijau untuk rasm/badan huruf
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

CURVE_FITTING_DIR, CURVE_FITTING_PER_HURUF_DIR, CURVE_FITTING_CSV_DIR = make_curve_fitting_output_dirs(imagename)
print(f"[CURVE FITTING] Folder output utama: {CURVE_FITTING_DIR}")
print(f"[CURVE FITTING] Folder output per huruf: {CURVE_FITTING_PER_HURUF_DIR}")
print(f"[CURVE FITTING] Folder output CSV: {CURVE_FITTING_CSV_DIR}")

# CSV Bezier juga diarahkan ke folder curve fitting/csv
BEZIER_CSV_DIR = CURVE_FITTING_CSV_DIR if SAVE_BEZIER_OUTPUT_CSV else None
if SAVE_BEZIER_OUTPUT_CSV:
    print(f"[BEZIER CSV] Folder output CSV: {BEZIER_CSV_DIR}")

if PRINT_BEZIER_REFERENCE_DEMO:
    _xx_bezier_reference_demo = run_bezier_reference_demo_output(
        csv_dir=BEZIER_CSV_DIR,
        image_prefix=os.path.basename(imagename)
    )

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
            scribe.add_node(int(filled), label=int(lbls[cy,cx]), area=(len(moments[n])-1)/pow(SLIC_SPACE,2), hurf='', pos_bitmap=(cx,cy), pos_render=(cx,-cy), color='#00FF00', rasm=True)
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
circular_blob_loop_components = []   # skeleton loop dari circular_blob/DT-loop untuk Bezier forced closed

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
        "diacritic": (0, 0, 255),       # merah untuk diakritik (BGR/OpenCV)
        "circular_blob": (0, 180, 0),   # hijau untuk rasm/blob
        "normal": (0, 180, 0),          # hijau untuk rasm/badan huruf
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

    # Simpan mask loop yang berasal dari circular_blob/DT-loop.
    # Mask ini nanti dipakai untuk memaksa Bezier membuat closed loop.
    _forced_loop_for_bezier = np.zeros_like(skel_hybrid, dtype=np.uint8)
    if blob_type == 'circular_blob':
        _forced_loop_for_bezier |= (skel_hybrid > 0).astype(np.uint8)
    if 'loop_added' in locals() and np.any(loop_added > 0):
        _forced_loop_for_bezier |= (loop_added > 0).astype(np.uint8)
    if np.any(_forced_loop_for_bezier > 0):
        circular_blob_loop_components.append(_forced_loop_for_bezier.copy())

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

# Mask forced-loop untuk Bezier: diambil dari skeleton hybrid circular_blob/DT-loop.
# Ini BUKAN contour dari binary, melainkan piksel skeleton yang sudah dibentuk
# oleh Hybrid Zhang-Suen + Distance Transform + Circular Blob Detection.
CIRCULAR_BLOB_LOOP_SKELETON = np.zeros_like(combined_skeleton, dtype=np.uint8)
try:
    for _loop_skel in circular_blob_loop_components:
        CIRCULAR_BLOB_LOOP_SKELETON |= (_loop_skel > 0).astype(np.uint8)
except Exception:
    pass
CIRCULAR_BLOB_LOOP_SKELETON = ((CIRCULAR_BLOB_LOOP_SKELETON > 0) & (combined_skeleton > 0)).astype(np.uint8) * 255
print(f"[BEZIER LOOP] forced circular/blob loop skeleton points = {int(np.count_nonzero(CIRCULAR_BLOB_LOOP_SKELETON))}")


# ============================================================
# TAMBAHAN: sumber skeleton untuk Bezier dari huruf yang sudah dipotong
# ============================================================
BEZIER_SIG_SOURCE_SKELETON = combined_skeleton.copy() if combined_skeleton is not None else None
BEZIER_SIG_SOURCE_LABELS = None
BEZIER_SIG_SOURCE_INFO = None

if USE_ARABIC_LETTER_CUT_FOR_BEZIER:
    try:
        print("\n" + "=" * 60)
        print("ARABIC LETTER CUT AKTIF UNTUK SUMBER BEZIER/CURVE FITTING")
        print("Bezier akan mengambil skeleton dari huruf tunggal hasil potong.")
        print("=" * 60)
        BEZIER_SIG_SOURCE_SKELETON, BEZIER_SIG_SOURCE_LABELS, BEZIER_SIG_SOURCE_INFO = make_bezier_skeleton_source_from_cut_letters(
            binary_img=gray,
            baseline_y=baseline_y,
            median_body_h=median_body_h,
            r_stroke_global=r_stroke_global,
            imagename=imagename,
            include_marks_in_bezier=LETTER_CUT_INCLUDE_MARKS_IN_BEZIER,
            debug=True,
            base_skeleton_for_bezier=combined_skeleton,
            forced_loop_skeleton=CIRCULAR_BLOB_LOOP_SKELETON,
        )
        if BEZIER_SIG_SOURCE_SKELETON is None or not np.any(BEZIER_SIG_SOURCE_SKELETON > 0):
            print("[ARABIC LETTER CUT] Source kosong, fallback ke combined_skeleton lama.")
            BEZIER_SIG_SOURCE_SKELETON = combined_skeleton.copy()
            BEZIER_SIG_SOURCE_LABELS = None
    except Exception as e:
        print(f"[ARABIC LETTER CUT] Gagal membuat source huruf tunggal: {e}")
        print("[ARABIC LETTER CUT] Fallback ke combined_skeleton lama.")
        BEZIER_SIG_SOURCE_SKELETON = combined_skeleton.copy() if combined_skeleton is not None else None
        BEZIER_SIG_SOURCE_LABELS = None

# Forced-loop mask yang sudah selaras dengan source huruf hasil cut.
BEZIER_FORCED_LOOP_MASK = np.zeros_like(BEZIER_SIG_SOURCE_SKELETON, dtype=np.uint8) if BEZIER_SIG_SOURCE_SKELETON is not None else None
if BEZIER_FORCED_LOOP_MASK is not None:
    BEZIER_FORCED_LOOP_MASK[((CIRCULAR_BLOB_LOOP_SKELETON > 0) & (BEZIER_SIG_SOURCE_SKELETON > 0))] = 255
    print(f"[BEZIER LOOP] forced loop points pada source huruf cut = {int(np.count_nonzero(BEZIER_FORCED_LOOP_MASK))}")

# ============================================================
# VISUALISASI SKELETON BERWARNA
# Rasm/badan huruf dibuat HIJAU, sedangkan diakritik dibuat MERAH.
#
# Revisi v2:
# - Versi sebelumnya hanya mengandalkan label `diacritic` dari classifier.
#   Pada beberapa citra, titik/diakritik ikut terklasifikasi sebagai `normal`,
#   sehingga semua skeleton tampak hijau.
# - Sekarang mask merah ditambah dengan aturan jarak: komponen kecil yang
#   TERPISAH dan JAUH dari rasm/body utama juga dianggap diakritik.
# ============================================================
def build_far_diacritic_mask_from_rasm(binary_img, skeleton_img, r_stroke=1.5, debug=True):
    """
    Deteksi titik/diakritik untuk pewarnaan skeleton.

    Prinsip:
    1) Cari body/rasm utama dari binary menggunakan komponen yang cukup besar,
       tinggi, atau lebar.
    2) Hitung jarak setiap komponen kecil terhadap body/rasm.
    3) Jika komponen kecil berada cukup jauh dari body/rasm, skeleton pada
       komponen itu diberi mask merah.

    Fungsi ini hanya memengaruhi WARNA visualisasi, bukan bentuk skeleton.
    """
    bw = (binary_img > 0).astype(np.uint8)
    sk = (skeleton_img > 0).astype(np.uint8)
    out_mask = np.zeros_like(sk, dtype=np.uint8)

    if bw.sum() == 0 or sk.sum() == 0:
        return out_mask.astype(bool)

    H2, W2 = bw.shape

    # Ambang body/rasm mengikuti logika body mask pada script, tetapi dibuat
    # sedikit adaptif terhadap ukuran gambar.
    body_area_min = max(220, int(18.0 * (float(r_stroke) ** 2)))
    body_h_min = max(18, int(round(7.0 * float(r_stroke))))
    body_w_min = max(24, int(0.035 * W2))

    # Komponen kandidat diakritik biasanya kecil. Nilai ini sengaja longgar,
    # lalu tetap divalidasi dengan jarak dari rasm.
    diac_area_max = max(350, int(90.0 * (float(r_stroke) ** 2)))
    diac_dist_min = max(4.0, 2.0 * float(r_stroke))

    n, labels, stats, cents = cv.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        return out_mask.astype(bool)

    body_mask = np.zeros_like(bw, dtype=np.uint8)
    body_areas = []
    body_heights = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        is_body = (area >= body_area_min) or (h >= body_h_min) or (w >= body_w_min)
        if is_body:
            body_mask[labels == i] = 1
            body_areas.append(int(area))
            body_heights.append(int(h))

    # Fallback: kalau semua komponen terlalu kecil, jadikan komponen terbesar
    # sebagai body supaya distance transform tetap punya rujukan rasm.
    if body_mask.sum() == 0:
        largest_id = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        body_mask[labels == largest_id] = 1
        body_areas.append(int(stats[largest_id, cv.CC_STAT_AREA]))
        body_heights.append(int(stats[largest_id, cv.CC_STAT_HEIGHT]))

    # Dilasi tipis agar fragmen skeleton yang sangat dekat dengan body tidak
    # salah diberi warna merah.
    k_body = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    body_near = cv.dilate(body_mask, k_body, iterations=1) > 0
    dt_to_body = distance_transform_edt(~body_near)

    max_body_area = max(body_areas) if body_areas else 1
    med_body_h = float(np.median(body_heights)) if body_heights else float(body_h_min)

    # Pass 1: gunakan komponen binary. Ini paling akurat karena titik/diakritik
    # biasanya komponen terpisah pada binary sebelum skeleton.
    picked_binary = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        comp = (labels == i)

        # Lewati body/rasm utama.
        if np.any(body_mask[comp] > 0):
            continue

        comp_skel = (sk > 0) & comp
        if not np.any(comp_skel):
            # beri toleransi 1px karena thinning bisa sedikit bergeser
            comp_dil = cv.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            comp_skel = (sk > 0) & comp_dil
        if not np.any(comp_skel):
            continue

        dist_med = float(np.median(dt_to_body[comp_skel]))
        small_enough = (area <= diac_area_max) or (area <= 0.18 * max_body_area and h <= 0.85 * med_body_h)
        far_enough = dist_med >= diac_dist_min

        if small_enough and far_enough:
            out_mask[comp_skel] = 1
            picked_binary += 1

    # Pass 2: fallback langsung dari connected-component skeleton.
    # Ini menangkap titik yang pada binary sudah menyatu tipis, tetapi pada
    # skeleton tetap menjadi komponen kecil yang jauh dari body.
    nsk, sk_labels, sk_stats, _ = cv.connectedComponentsWithStats(sk, connectivity=8)
    if nsk > 1:
        sk_areas = sk_stats[1:, cv.CC_STAT_AREA]
        largest_skel_area = int(sk_areas.max()) if sk_areas.size else 1
        for i in range(1, nsk):
            area = int(sk_stats[i, cv.CC_STAT_AREA])
            if area <= 0:
                continue
            comp_skel = (sk_labels == i)
            if np.any(body_near & comp_skel):
                continue
            dist_med = float(np.median(dt_to_body[comp_skel]))
            small_skel = area <= max(6, int(0.08 * largest_skel_area))
            if small_skel and dist_med >= diac_dist_min:
                out_mask[comp_skel] = 1

    if debug:
        print(
            f"[WARNA DIAKRITIK] area_body_min={body_area_min}, h_body_min={body_h_min}, "
            f"w_body_min={body_w_min}, area_diac_max={diac_area_max}, "
            f"dist_min={diac_dist_min:.1f}px, komponen_binary_merah={picked_binary}, "
            f"piksel_skeleton_merah={int(out_mask.sum())}"
        )

    return (out_mask > 0)


combined_diacritic_mask = np.zeros((H, W), dtype=np.uint8)

# Sumber 1: komponen yang memang berhasil diklasifikasikan sebagai diacritic.
for _skel, _ctype in zip(skeleton_components_hybrid, component_types):
    if _ctype == 'diacritic':
        _skel_diac = (_skel > 0) & (~blob_wall)
        combined_diacritic_mask |= _skel_diac.astype(np.uint8)

# Sumber 2: fallback jarak dari rasm/body. Ini yang membuat titik yang jauh
# dari rasm menjadi merah walaupun classifier awal tidak menamainya diacritic.
far_diacritic_mask = build_far_diacritic_mask_from_rasm(
    binary_img=gray,
    skeleton_img=combined_skeleton,
    r_stroke=r_stroke_global,
    debug=True
)
combined_diacritic_mask |= far_diacritic_mask.astype(np.uint8)

combined_diacritic_mask = (combined_diacritic_mask > 0) & (combined_skeleton > 0)
combined_skeleton_color = np.zeros((H, W, 3), dtype=np.uint8)
combined_skeleton_color[combined_skeleton > 0] = SKELETON_RASM_RGB
combined_skeleton_color[combined_diacritic_mask] = SKELETON_DIACRITIC_RGB

colored_skeleton_path = f"{imagename}_skeleton_rasm_hijau_diacritic_merah.png"
cv.imwrite(colored_skeleton_path, cv.cvtColor(combined_skeleton_color, cv.COLOR_RGB2BGR))
print(f"Skeleton berwarna disimpan: {colored_skeleton_path}")
print(f"Jumlah piksel skeleton diakritik/merah: {int(combined_diacritic_mask.sum())}")
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
    plt.imshow(combined_skeleton_color)
    plt.title("Hybrid Skeleton Zhang-Suen Distance Transform + Circular Blob Detection")
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
plt.imshow(combined_skeleton_color, alpha=(combined_skeleton > 0).astype(np.float32) * 0.85)
plt.title("DT map + Zhang-Suen Skeleton overlay berwarna\nRasm=hijau, Diakritik=merah")
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


# SIGCurveReconstruction dan gaussian smoothing dihapus dari pipeline Bezier.
# Bezier di bawah memakai titik skeleton hybrid langsung.


# BEZIER PER HURUF LANGSUNG DARI SKELETON HYBRID HASIL CUT
# SIG Curve Reconstruction dihapus dari pipeline ini.
# ============================================================
print("\n" + "="*60)
print("BEZIER LANGSUNG DARI SKELETON HYBRID HASIL CUT")
print("Sumber titik: skeleton hybrid Zhang-Suen + DT + circular blob detection")
print("="*60)

sig_source_skeleton = BEZIER_SIG_SOURCE_SKELETON if 'BEZIER_SIG_SOURCE_SKELETON' in globals() else combined_skeleton
sig_source_labels = BEZIER_SIG_SOURCE_LABELS if 'BEZIER_SIG_SOURCE_LABELS' in globals() else None
forced_loop_mask_for_bezier = BEZIER_FORCED_LOOP_MASK if 'BEZIER_FORCED_LOOP_MASK' in globals() else None

bezier_curves = {}
SHOW_BEZIER_OVERLAY = False
SHOW_SKELETON_POINTS = True

if sig_source_skeleton is not None and np.any(sig_source_skeleton > 0):
    if sig_source_labels is not None and np.any(sig_source_labels > 0):
        labels_sig = sig_source_labels.astype(np.int32)
        num_labels_sig = int(labels_sig.max()) + 1
        print(f"Jumlah huruf hasil cut untuk sumber Bezier: {num_labels_sig - 1}")
    else:
        num_labels_sig, labels_sig = cv.connectedComponents(
            (sig_source_skeleton > 0).astype(np.uint8), connectivity=8
        )
        print(f"Jumlah komponen skeleton hybrid: {num_labels_sig - 1}")

    comp_leftmost = {}
    for comp_id in range(1, num_labels_sig):
        ys_c, xs_c = np.where(labels_sig == comp_id)
        if len(xs_c) > 0:
            comp_leftmost[comp_id] = int(xs_c.min())
    comp_order = sorted(comp_leftmost.keys(), key=lambda c: comp_leftmost[c])

    print("\n" + "="*60)
    print("BEZIER CURVE PER HURUF DARI SKELETON HYBRID HURUF HASIL CUT")
    print("="*60)

    prefix = os.path.basename(imagename)
    bezier_dir = CURVE_FITTING_PER_HURUF_DIR
    os.makedirs(bezier_dir, exist_ok=True)
    bezier_csv_summary_rows = []
    curve_fitting_summary_rows = []

    for disp_idx, cv_id in enumerate(comp_order, start=1):
        mask_comp = ((labels_sig == cv_id).astype(np.uint8) * 255)
        n_pts = int((mask_comp > 0).sum())
        x_left = comp_leftmost[cv_id]
        if n_pts < 2:
            print(f"  Huruf {disp_idx}: skip, titik skeleton kurang.")
            continue

        try:
            subpaths = build_bezier_subpaths_from_skeleton_mask(
                mask_comp,
                ordered_points_fallback=None,
                sigma=1.5,
                min_open_points=2,
                forced_loop_mask=forced_loop_mask_for_bezier,
            )
            if not subpaths:
                print(f"  Huruf {disp_idx}: skip, tidak ada subpath skeleton valid.")
                continue

            subcurves = []
            sub_skeletons = []
            all_controls = []
            subpath_infos = []

            for sp_idx, sp in enumerate(subpaths, start=1):
                sp_pts = np.asarray(sp["points"], dtype=float)
                if len(sp_pts) < 2:
                    continue

                sp_curve, sp_controls, sp_closed = make_bezier_curve_per_letter_from_ordered_skeleton(
                    sp_pts,
                    mask_comp=None,
                    samples_per_segment=14,
                    force_closed=bool(sp["closed"])
                )
                if len(sp_curve) < 2:
                    continue

                subcurves.append(np.asarray(sp_curve, dtype=float))
                sub_skeletons.append(np.asarray(sp_pts, dtype=float))
                all_controls.extend(sp_controls)
                subpath_infos.append({
                    "kind": sp.get("kind", "unknown"),
                    "closed": bool(sp_closed),
                    "skeleton_points": int(len(sp_pts)),
                    "curve_points": int(len(sp_curve)),
                    "segment_count": int(len(sp_controls)),
                })

            if not subcurves:
                print(f"  Huruf {disp_idx}: skip, semua subpath gagal dibuat kurva.")
                continue

            ys_eval, xs_eval = np.where(mask_comp > 0)
            eval_skeleton_pts = np.column_stack([xs_eval, ys_eval]).astype(float)
            if len(eval_skeleton_pts) == 0:
                eval_skeleton_pts = np.vstack(sub_skeletons)

            curve_pts = np.vstack(subcurves)
            closed_flag = any(info["closed"] for info in subpath_infos)
            fit_eval = evaluate_curve_fitting_to_skeleton(eval_skeleton_pts, curve_pts)

            distance_csv_file = ""
            if CURVE_FITTING_CSV_DIR is not None:
                distance_csv = os.path.join(
                    CURVE_FITTING_CSV_DIR,
                    f"{prefix}_huruf_{disp_idx:03d}_curve_fitting_distance.csv"
                )
                save_curve_fitting_distance_csv(
                    distance_csv,
                    eval_skeleton_pts,
                    curve_pts,
                    huruf=disp_idx,
                )
                distance_csv_file = os.path.basename(distance_csv)

            curve_fitting_summary_rows.append(
                build_curve_fitting_summary_row(
                    huruf=disp_idx,
                    skeleton_pts=eval_skeleton_pts,
                    curve_pts=curve_pts,
                    control_segments=all_controls,
                    closed_flag=closed_flag,
                    cv_id=cv_id,
                    x_left=x_left,
                    fit_eval=fit_eval,
                    distance_csv_file=distance_csv_file,
                )
            )

            bezier_curves[disp_idx] = {
                "curve": np.asarray(curve_pts, dtype=float),
                "skeleton": np.asarray(eval_skeleton_pts, dtype=float),
                "subcurves": subcurves,
                "sub_skeletons": sub_skeletons,
                "subpath_infos": subpath_infos,
                "controls": all_controls,
                "cv_id": cv_id,
                "x_left": x_left,
                "closed": closed_flag,
                "fit_eval": fit_eval,
                "distance_csv_file": distance_csv_file,
            }

            forced_loop_count = sum(1 for info in subpath_infos if info.get("kind") == "forced_circular_blob_loop")
            loop_count = sum(1 for info in subpath_infos if info["closed"])
            open_count = len(subpath_infos) - loop_count
            print(f"  Huruf {disp_idx}: OK | subpath={len(subpath_infos)} "
                  f"(forced_loop={forced_loop_count}, loop={loop_count}, open={open_count}) | "
                  f"skeleton={len(eval_skeleton_pts)} titik | "
                  f"segmen={len(all_controls)} | curve={len(curve_pts)} titik | "
                  f"closed_any={closed_flag} | RMSE={fit_eval['rmse_skeleton_to_curve']:.4f} px | "
                  f"Hausdorff={fit_eval['hausdorff_symmetric']:.4f} px")

            tt_huruf = np.linspace(0, 1, len(curve_pts))

            if PRINT_BEZIER_PER_HURUF_CALCULATION_OUTPUT:
                print(f"\n[BEZIER PER-HURUF] Huruf {disp_idx} - output perhitungan")
                print_bezier_output_like_reference(tt_huruf, curve_pts, curve_pts)

            if SAVE_BEZIER_OUTPUT_CSV and BEZIER_CSV_DIR is not None:
                huruf_csv = os.path.join(
                    BEZIER_CSV_DIR,
                    f"{prefix}_huruf_{disp_idx:03d}_bezier_output.csv"
                )
                save_bezier_points_csv(
                    huruf_csv,
                    tt_huruf,
                    curve_pts,
                    curve_pts,
                    label="per_huruf_hybrid_skeleton_forced_loop_no_sig",
                    huruf=disp_idx,
                )
                bezier_csv_summary_rows.append(
                    build_bezier_summary_row(
                        "per_huruf_hybrid_skeleton_forced_loop_no_sig",
                        tt_huruf,
                        curve_pts,
                        curve_pts,
                        huruf=disp_idx,
                        skeleton_points=int(len(eval_skeleton_pts)),
                        segment_count=int(len(all_controls)),
                        curve_points=int(len(curve_pts)),
                        subpath_count=int(len(subpath_infos)),
                        forced_loop_count=int(forced_loop_count),
                        closed_loop_count=int(loop_count),
                        open_path_count=int(open_count),
                        closed=bool(closed_flag),
                        cv_label=int(cv_id),
                        x_left=float(x_left),
                        csv_file=os.path.basename(huruf_csv),
                        distance_csv_file=distance_csv_file,
                        rmse_skeleton_to_curve=fit_eval["rmse_skeleton_to_curve"],
                        mae_skeleton_to_curve=fit_eval["mae_skeleton_to_curve"],
                        sse_skeleton_to_curve=fit_eval["sse_skeleton_to_curve"],
                        hausdorff_symmetric=fit_eval["hausdorff_symmetric"],
                    )
                )
                print(f"[BEZIER CSV] Huruf {disp_idx} disimpan: {huruf_csv}")

        except Exception as e:
            print(f"  Huruf {disp_idx}: gagal Bezier -> {e}")
            continue

    valid_keys = sorted(bezier_curves.keys())
    if len(valid_keys) > 0:
        ncols = min(8, len(valid_keys))
        nrows = int(math.ceil(len(valid_keys) / ncols))
        fig_grid, axes_grid = plt.subplots(
            nrows, ncols,
            figsize=(2.15 * ncols, 3.05 * nrows),
            squeeze=False
        )
        axes_flat = axes_grid.flatten()

        for plot_idx, disp_idx in enumerate(valid_keys):
            ax = axes_flat[plot_idx]
            item = bezier_curves[disp_idx]
            curve_pts = item["curve"]
            skel_pts = item["skeleton"]
            subcurves = item.get("subcurves", [curve_pts])
            sub_skeletons = item.get("sub_skeletons", [skel_pts])
            subpath_infos = item.get("subpath_infos", [])

            for sc in subcurves:
                sc = np.asarray(sc, dtype=float)
                if len(sc) >= 2:
                    ax.plot(sc[:, 0], sc[:, 1], '-', linewidth=1.1)

            if SHOW_SKELETON_POINTS:
                for ss in sub_skeletons:
                    ss = np.asarray(ss, dtype=float)
                    if len(ss) >= 1:
                        ax.scatter(ss[:, 0], ss[:, 1], s=3, alpha=0.22)

            loop_count = sum(1 for info in subpath_infos if info.get("closed", False))
            forced_loop_count = sum(1 for info in subpath_infos if info.get("kind") == "forced_circular_blob_loop")
            ax.set_title(f"Huruf {disp_idx} | loop={loop_count} | forced={forced_loop_count}", fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()
            ax.grid(False)

            all_pts = np.vstack([curve_pts, skel_pts])
            xmin, ymin = np.min(all_pts, axis=0)
            xmax, ymax = np.max(all_pts, axis=0)
            pad_x = max(4.0, 0.08 * (xmax - xmin + 1.0))
            pad_y = max(4.0, 0.08 * (ymax - ymin + 1.0))
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymax + pad_y, ymin - pad_y)

        for idx in range(len(valid_keys), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig_grid.suptitle(
            "Kurva Bezier per Huruf dari Skeleton Hybrid Hasil Cut",
            fontsize=12
        )
        fig_grid.text(
            0.5, 0.015,
            "Loop circular_blob dipaksa menjadi closed sub-path dari skeleton hybrid; SIG curve tidak dipakai.",
            ha='center', fontsize=8
        )
        plt.tight_layout(rect=[0, 0.035, 1, 0.95])
        bezier_grid_out = os.path.join(CURVE_FITTING_DIR, f"{prefix}_bezier_perhuruf_grid.png")
        plt.savefig(bezier_grid_out, dpi=200)
        plt.show()
        print(f"\n[BEZIER] Output grid per huruf disimpan: {bezier_grid_out}")

        # Simpan plot individual per huruf. Tiap subpath dipisah agar tidak ada
        # garis palsu antara loop dan open stroke.
        for disp_idx in valid_keys:
            item = bezier_curves[disp_idx]
            curve_pts = item["curve"]
            skel_pts = item["skeleton"]

            fig_one, ax_one = plt.subplots(figsize=(4, 4))
            for sc in item.get("subcurves", [curve_pts]):
                sc = np.asarray(sc, dtype=float)
                if len(sc) >= 2:
                    ax_one.plot(sc[:, 0], sc[:, 1], '-', linewidth=1.4)
            if SHOW_SKELETON_POINTS:
                for ss in item.get("sub_skeletons", [skel_pts]):
                    ss = np.asarray(ss, dtype=float)
                    if len(ss) >= 1:
                        ax_one.scatter(ss[:, 0], ss[:, 1], s=4, alpha=0.22)
            ax_one.set_aspect('equal', adjustable='box')
            ax_one.invert_yaxis()
            fit_eval = item.get("fit_eval", {})
            rmse_title = fit_eval.get("rmse_skeleton_to_curve", 0.0)
            hausdorff_title = fit_eval.get("hausdorff_symmetric", 0.0)
            loop_count = sum(1 for info in item.get("subpath_infos", []) if info.get("closed", False))
            forced_loop_count = sum(1 for info in item.get("subpath_infos", []) if info.get("kind") == "forced_circular_blob_loop")
            ax_one.set_title(
                f"Huruf {disp_idx} | loop={loop_count} | forced={forced_loop_count}\nRMSE={rmse_title:.4f}px | Hausdorff={hausdorff_title:.4f}px",
                fontsize=9
            )
            ax_one.grid(False)
            one_out = os.path.join(bezier_dir, f"huruf_{disp_idx:03d}_bezier.png")
            plt.tight_layout()
            plt.savefig(one_out, dpi=200)
            plt.close(fig_one)

        print(f"[BEZIER] Output per huruf disimpan di folder: {bezier_dir}/")

        if CURVE_FITTING_CSV_DIR is not None and curve_fitting_summary_rows:
            curve_fit_summary_csv = os.path.join(
                CURVE_FITTING_CSV_DIR,
                f"{prefix}_curve_fitting_summary.csv"
            )
            _write_rows_csv(curve_fit_summary_csv, curve_fitting_summary_rows)
            print(f"[CURVE FITTING CSV] Ringkasan error fitting disimpan: {curve_fit_summary_csv}")

        if SAVE_BEZIER_OUTPUT_CSV and BEZIER_CSV_DIR is not None and bezier_csv_summary_rows:
            summary_csv = os.path.join(BEZIER_CSV_DIR, f"{prefix}_bezier_perhuruf_summary.csv")
            _write_rows_csv(summary_csv, bezier_csv_summary_rows)
            print(f"[BEZIER CSV] Ringkasan semua huruf disimpan: {summary_csv}")

    if SHOW_BEZIER_OVERLAY and len(bezier_curves) > 0:
        fig_bz, ax_bz = plt.subplots(figsize=(12, 5))
        bg_show_bz = gray if len(gray.shape) == 2 else cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        ax_bz.imshow(bg_show_bz, cmap='gray', alpha=0.22)
        skel_y_bz, skel_x_bz = np.where(sig_source_skeleton > 0)
        ax_bz.scatter(skel_x_bz, skel_y_bz, c='cyan', s=1, alpha=0.20, label='Skeleton')

        cmap_bz = plt.cm.tab20(np.linspace(0, 1, max(len(bezier_curves), 1)))
        for i, disp_idx in enumerate(sorted(bezier_curves.keys())):
            item_bz = bezier_curves[disp_idx]
            color = cmap_bz[i % len(cmap_bz)]
            first_label = True
            for sc in item_bz.get("subcurves", [item_bz["curve"]]):
                sc = np.asarray(sc, dtype=float)
                if len(sc) < 2:
                    continue
                ax_bz.plot(
                    sc[:, 0], sc[:, 1], '-', color=color,
                    linewidth=2.0,
                    label=f'Huruf-{disp_idx}' if first_label else None
                )
                first_label = False

        ax_bz.set_title("Overlay Kurva Bezier per Huruf dari Skeleton Hybrid Hasil Cut")
        ax_bz.axis('off')
        if len(bezier_curves) <= 12:
            ax_bz.legend(loc='upper right', fontsize=6, markerscale=2, ncol=2)
        plt.tight_layout()
        bezier_out = os.path.join(CURVE_FITTING_DIR, f"{prefix}_bezier_perhuruf_overlay.png")
        plt.savefig(bezier_out, dpi=200)
        plt.show()
        print(f"[BEZIER] Output overlay gabungan disimpan: {bezier_out}")

    print(f"[BEZIER] Total komponen Bezier: {len(bezier_curves)}")
else:
    print("Skeleton sumber Bezier kosong, Bezier tidak dapat dijalankan.")


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
