# SCRIPT BERHASIL MEMOTONG HURUF SESUAI DENGAN STUKTUR HURUFNYA PER SUB-PATH





# usage: python -u line2hist.py <inputimage>
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
        
filename= sys.argv[1]
#filename= 'topanribut.png'
imagename, ext= os.path.splitext(filename)
image = cv.imread(filename)
resz = cv.resize(image, (RESIZE_FACTOR*image.shape[1], RESIZE_FACTOR*image.shape[0]), interpolation=cv.INTER_LINEAR)
image= resz.copy()
image=  cv.bitwise_not(image)
height= image.shape[0]
width= image.shape[1]

image_gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# image_gray= image[:,:,CHANNEL]
_, gray = cv.threshold(image_gray, 0, THREVAL, cv.THRESH_OTSU)
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
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
plt.title("Thresholded (Otsu)")
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
    cv.floodFill(ccv, None, seed, RASMVAL, loDiff=(5), upDiff=(5))
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



# SKELETON OLD
skeleton_components = []

for n in range(len(components)):
    binary_mat = (components[n].mat == RASMVAL).astype(np.uint8)
    skeleton = skeletonize(binary_mat)
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
    plt.title("Skeletonization")
    plt.axis('off')
    plt.show()
else:
    print("Skeleton kosong, tidak ada data untuk ditampilkan.")
    

# DETEKSI TEPI (EDGE DETECTION)
edges = cv.Canny(gray, 30, 30)

# Temukan kontur berdasarkan gambar hasil deteksi tepi
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Opsional: tampilkan hasil deteksi tepi untuk verifikasi
plt.imshow(edges, cmap="gray")
plt.title("Edges")
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









# FREEMAN CHAIN CODE
#    3   2   1
#      \ | /
#    4 ------0
#      / | \
#    5   6   7
def direction_code(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    directions = {
        (1, 0): 0, # kanan
        (1, -1): 1, #kanan atas
        (0, -1): 2, #atas
        (-1, -1): 3, #kiri atas
        (-1, 0): 4, #kiri
        (-1, 1): 5, #kiri bawah
        (0, 1): 6, #bawah
        (1, 1): 7 #kanan bawah
    }
    return directions.get((dx, dy), -1)  # -1 jika bukan tetangga langsung










# TRAVELLING SALESMAN PROBLEM (TSP)
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import euclidean
from collections import defaultdict
from skimage.morphology import skeletonize

# ====================
# PARAMETER
# ====================
JARAK_MAKSIMUM = 4  
LOOP_RADIUS = 6
JARAK_PEMISAH_HURUF = 12
BATAS_CLUSTER_X = 20   # ← ini kunci sebenarnya



# ====================
# HELPER FUNCTIONS
# ====================

def nearest_endpoint_to_node(G, node):
    """
    Ambil endpoint (degree==1) yang paling dekat (jarak euclidean) ke node tertentu.
    Kalau tidak ada endpoint, balikin node itu sendiri.
    """
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if not endpoints:
        return node
    return min(endpoints, key=lambda e: euclidean(e, node))


def farthest_endpoint_from_start(G, start_node):
    """
    Ambil endpoint yang paling jauh dari start_node berdasarkan shortest path di graph.
    Kalau graph putus / tidak ada endpoint, fallback pakai euclidean.
    """
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if not endpoints:
        return start_node

    # shortest path length dari start ke semua node
    dist = nx.single_source_shortest_path_length(G, start_node)

    # endpoint terjauh yg reachable
    reachable_endpoints = [e for e in endpoints if e in dist]
    if reachable_endpoints:
        return max(reachable_endpoints, key=lambda e: dist[e])

    # fallback (kalau tidak reachable)
    return max(endpoints, key=lambda e: euclidean(e, start_node))



def direction_code(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    directions = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    return directions.get((dx, dy), -1)

def build_skeleton_graph(skeleton_img):
    h, w = skeleton_img.shape
    G = nx.Graph()
    coords = np.column_stack(np.where(skeleton_img > 0))
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skeleton_img[ny, nx_] > 0:
                    p1 = (x, y)
                    p2 = (nx_, ny)
                    dist = euclidean(p1, p2)
                    G.add_edge(p1, p2, weight=dist)
    return G

def tsp_greedy_with_revisit(points, branch_nodes_idx, max_visits=2, start_point=None, end_point=None):
    """
    Greedy TSP dengan aturan:
    - Node NON-branch hanya boleh dikunjungi 1x.
    - Node branch (degree>=3) boleh dikunjungi sampai max_visits.
    - start_point: titik mulai (kalau tidak persis ada, snap ke titik terdekat).
    - end_point: titik akhir 'ditahan' sampai langkah terakhir (baru dipakai di akhir).
    """
    if not points:
        return []

    pts = list(points)
    idx_of = {p: i for i, p in enumerate(pts)}

    def snap_to_index(p):
        if p is None:
            return None
        if p in idx_of:
            return idx_of[p]
        # snap ke titik terdekat (euclidean)
        return min(range(len(pts)), key=lambda i: euclidean(pts[i], p))

    start_idx = snap_to_index(start_point)
    end_idx   = snap_to_index(end_point)

    # default start: paling kanan (x terbesar)
    if start_idx is None:
        start_idx = int(np.argmax([p[0] for p in pts]))

    # kalau start==end, anggap tidak ada end khusus
    if end_idx is not None and end_idx == start_idx:
        end_idx = None

    branch_set = set(branch_nodes_idx)
    visits = [0] * len(pts)

    def limit(i):
        return max_visits if i in branch_set else 1

    current = start_idx
    tsp_path = [pts[current]]
    visits[current] += 1

    # semua node (kecuali end_idx) harus dikunjungi
    remaining = set(range(len(pts)))
    if end_idx is not None and end_idx in remaining:
        remaining.remove(end_idx)

    # tandai start sudah visited (kalau start bukan end)
    if current in remaining:
        remaining.remove(current)

    while remaining:
        cur_pt = pts[current]
        best_i = None
        best_d = float("inf")

        for i in remaining:
            if visits[i] >= limit(i):
                continue
            d = euclidean(cur_pt, pts[i])
            if d < best_d:
                best_d = d
                best_i = i

        if best_i is None:
            break

        visits[best_i] += 1
        tsp_path.append(pts[best_i])
        current = best_i
        remaining.remove(best_i)

    # terakhir baru masukkan end_idx kalau ada
    if end_idx is not None:
        if visits[end_idx] < limit(end_idx):
            tsp_path.append(pts[end_idx])
            visits[end_idx] += 1

    return tsp_path



def extract_branch_points(G, points):
    # titik dengan degree >= 3 dianggap percabangan
    return [i for i, p in enumerate(points) if p in G.nodes and G.degree[p] >= 3]


def densify_path_follow_graph(G, coarse_path):
    """
    Mengubah path yang lompat-lompat menjadi path rapat yang mengikuti skeleton graph.
    Tiap pasangan titik dihubungkan pakai shortest_path di graph.
    """
    if not coarse_path or len(coarse_path) < 2:
        return coarse_path

    dense = [coarse_path[0]]
    for a, b in zip(coarse_path, coarse_path[1:]):
        try:
            sp = nx.shortest_path(G, a, b, weight='weight')
            dense.extend(sp[1:])  # lanjutkan tanpa duplikasi titik awal
        except nx.NetworkXNoPath:
            # fallback: kalau gak ada jalur (harusnya jarang kalau satu huruf)
            dense.append(b)
    return dense

def nearest_endpoint_to_node(G, node):
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if not endpoints:
        return node
    dist = nx.single_source_shortest_path_length(G, node)
    return min(endpoints, key=lambda n: dist.get(n, 10**9))


def farthest_endpoint_from_start(G, start_node):
    endpoints = [n for n in G.nodes if G.degree[n] == 1]
    if not endpoints:
        return start_node
    dist = nx.single_source_shortest_path_length(G, start_node)
    return max(endpoints, key=lambda n: dist.get(n, -1))


def visualize_all_paths(skeleton, all_subpaths, visit_counts, branch_nodes):
    skeleton_rgb = cv.cvtColor((skeleton * 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
    plt.figure(figsize=(10, 10))
    plt.imshow(skeleton_rgb)
    total_distance = 0.0

    for path in all_subpaths:
        if len(path) < 2:
            continue

        # cari segmen utama yang benar-benar neighbor (biar dot tidak jadi end)
        segments = []
        seg = [0]
        for i in range(1, len(path)):
            if euclidean(path[i], path[i - 1]) <= 1.5:
                seg.append(i)
            else:
                segments.append(seg)
                seg = [i]
        segments.append(seg)
        main_seg = max(segments, key=len) if segments else list(range(len(path)))

        # gambar titik & garis
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            d = euclidean(p1, p2)

            # purple hanya kalau node itu branch & memang dikunjungi ulang
            visit_color = 'purple' if (visit_counts[p1] > 1 and p1 in branch_nodes) else 'red'
            plt.plot(p1[0], p1[1], 'o', color=visit_color, markersize=3)

            # GARIS hanya kalau tetangga (hindari garis nyasar)
            if d <= 1.5:
                total_distance += d
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1)

                code = direction_code(p1, p2)
                txt = str(code) if code != -1 else "?"
                plt.text(p1[0] + 1, p1[1], txt, color='white', fontsize=6)

        # Start/End di segmen utama
        sp = path[main_seg[0]]
        ep = path[main_seg[-1]]

        # Start = hijau
        plt.plot(sp[0], sp[1], 'go', markersize=8)
        plt.text(sp[0] + 2, sp[1], 'S', color='green', fontsize=9, weight='bold')

        # End = merah
        plt.plot(ep[0], ep[1], 'ro', markersize=8)
        plt.text(ep[0] + 2, ep[1], 'E', color='red', fontsize=9, weight='bold')

    plt.title(f"TSP per Huruf Arab\nTotal Distance: {total_distance:.2f}")
    plt.axis('off')
    plt.show()



# ====================
# ====================
# MAIN TSP OTOMATIS (jumlah huruf ikut input)
# ====================

MIN_DOT_SIZE = 25        # komponen lebih kecil dari ini dianggap dot/diakritik
MIN_LETTER_SIZE = 40     # minimal ukuran bagian agar layak jadi "huruf"
MIN_SPLIT_DX = 8         # minimal jarak centroid X agar dianggap pemisah huruf

def _centroid(nodes):
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    return (float(np.mean(xs)), float(np.mean(ys)))

def _best_bridge_split(G_sub):
    """Cari 1 bridge terbaik untuk memisahkan jadi 2 huruf."""
    best = None
    best_score = -1

    bridges = list(nx.bridges(G_sub))
    if not bridges:
        return None

    for (u, v) in bridges:
        H = G_sub.copy()
        if not H.has_edge(u, v):
            continue
        H.remove_edge(u, v)

        comps = [set(c) for c in nx.connected_components(H)]
        if len(comps) != 2:
            continue

        a, b = comps
        if min(len(a), len(b)) < MIN_LETTER_SIZE:
            continue

        ca = _centroid(a)
        cb = _centroid(b)
        dx = abs(ca[0] - cb[0])
        if dx < MIN_SPLIT_DX:
            continue

        score = min(len(a), len(b)) + 2.0 * dx
        if score > best_score:
            best_score = score
            best = (a, b, (u, v), score)

    return best

def _split_until_stable(G_total, nodes_set):
    """Split greedy sampai tidak ada bridge yang bagus lagi."""
    parts = [set(nodes_set)]
    cut_edges = []

    while True:
        best_global = None
        best_part_idx = None

        for i, part in enumerate(parts):
            Gp = G_total.subgraph(part).copy()
            res = _best_bridge_split(Gp)
            if res is None:
                continue
            if (best_global is None) or (res[3] > best_global[3]):
                best_global = res
                best_part_idx = i

        if best_global is None:
            break

        a, b, cut_edge, _ = best_global
        parts.pop(best_part_idx)
        parts.insert(best_part_idx, a)
        parts.insert(best_part_idx + 1, b)
        cut_edges.append(cut_edge)

    return parts, cut_edges


all_subpaths = []
visit_counts = defaultdict(int)
branch_nodes = set()

# 1) buat combined skeleton
combined_skeleton = None
for comp in components:
    binary_mat = (comp.mat == RASMVAL).astype(np.uint8)
    sk = skeletonize(binary_mat).astype(np.uint8)
    if combined_skeleton is None:
        combined_skeleton = np.zeros_like(sk, dtype=np.uint8)
    combined_skeleton |= sk

# 2) graph skeleton total
G_total = build_skeleton_graph(combined_skeleton)

# 3) CC total: pisah dot vs core
cc_list = [set(c) for c in nx.connected_components(G_total)]
cc_list.sort(key=len, reverse=True)

core_ccs = [c for c in cc_list if len(c) >= MIN_DOT_SIZE]
dot_ccs  = [c for c in cc_list if len(c) < MIN_DOT_SIZE]

# 4) split core jadi huruf otomatis
letter_groups = []
all_cut_edges = []

for core in core_ccs:
    parts, cuts = _split_until_stable(G_total, core)
    letter_groups.extend(parts)
    all_cut_edges.extend(cuts)

# kalau tidak ada huruf, stop
if not letter_groups:
    print("Tidak ada huruf terdeteksi.")
else:
    # 5) tempel dot ke huruf terdekat
    letter_centroids = [_centroid(g) for g in letter_groups]
    for dset in dot_ccs:
        cd = _centroid(dset)
        j = int(np.argmin([(cd[0]-c[0])**2 + (cd[1]-c[1])**2 for c in letter_centroids]))
        letter_groups[j] |= dset
        letter_centroids[j] = _centroid(letter_groups[j])

    # 6) urutkan huruf kanan->kiri
    letter_groups = sorted(letter_groups, key=lambda g: _centroid(g)[0], reverse=True)


# =========================
# ENTRY/EXIT NODE antar huruf
# =========================
# entry_node[i] = titik "masuk" huruf i (dari huruf kanan)
# exit_node[i]  = titik "keluar" huruf i (ke huruf kiri)

entry_node = {i: None for i in range(len(letter_groups))}
exit_node  = {i: None for i in range(len(letter_groups))}

BATAS_JUNCTION = JARAK_PEMISAH_HURUF  # boleh kamu kecilkan kalau terlalu gampang nyambung

# letter_groups diasumsikan sudah urut kanan->kiri (right-to-left)
for i in range(len(letter_groups) - 1):
    right_pts = np.array(list(letter_groups[i]), dtype=np.int32)
    left_pts  = np.array(list(letter_groups[i + 1]), dtype=np.int32)

    if len(right_pts) == 0 or len(left_pts) == 0:
        continue

    tree = cKDTree(left_pts)
    dists, idxs = tree.query(right_pts, k=1)

    j = int(np.argmin(dists))
    if float(dists[j]) <= BATAS_JUNCTION:
        pr = tuple(map(int, right_pts[j]))          # exit huruf kanan
        pl = tuple(map(int, left_pts[int(idxs[j])]))# entry huruf kiri
        exit_node[i] = pr
        entry_node[i + 1] = pl

    # Urutkan huruf dari kanan ke kiri berdasarkan centroid X
    letter_groups = sorted(
        letter_groups,
        key=lambda g: np.mean([p[0] for p in g]),
        reverse=True
    )

    # 7) buat mapping node -> group id
    node2gid = {}
    for gi, gset in enumerate(letter_groups):
        for n in gset:
            node2gid[n] = gi

    gx = [ _centroid(g)[0] for g in letter_groups ]

    # 8) tentukan exit junction (node batas ke huruf sebelah kiri)
    exit_node = {}  # key: group id, value: node

    for (u, v) in all_cut_edges:
        gu = node2gid.get(u, None)
        gv = node2gid.get(v, None)
        if gu is None or gv is None or gu == gv:
            continue

        # right group = centroid x lebih besar
        if gx[gu] > gx[gv]:
            right_gid = gu
            exit_candidate = u
        else:
            right_gid = gv
            exit_candidate = v

        # simpan exit yang paling kiri (x kecil) supaya benar-benar di junction
        if (right_gid not in exit_node) or (exit_candidate[0] < exit_node[right_gid][0]):
            exit_node[right_gid] = exit_candidate

    
   # 9) buat path per huruf
all_subpaths = []                 # ISI: body_path saja (buat plot + start/end + chain code)
all_letter_points = []            # ISI: titik huruf (body + dot) buat CROP
all_letters_nodes = []            # ISI: node huruf untuk CROPPING per huruf
all_dots = []                     # ISI: dots terpisah per huruf
visit_counts = defaultdict(int)
branch_nodes = set()
start_points = {}
end_points = {}

def pick_right_top(cands):
    """Pilih titik kanan-atas: x terbesar, y terkecil"""
    if not cands:
        return None
    return max(cands, key=lambda p: (p[0], -p[1]))

def pick_left_bottom(cands):
    """Pilih titik kiri-bawah: x terkecil, y terbesar"""
    if not cands:
        return None
    return min(cands, key=lambda p: (p[0], -p[1]))

for gi, gset in enumerate(letter_groups):
    Gg = G_total.subgraph(gset).copy()
    
    # pisah body vs dot (body = CC terbesar)
    subccs = [set(c) for c in nx.connected_components(Gg)]
    subccs.sort(key=len, reverse=True)
    body = subccs[0]
    dots = subccs[1:]
    
    G_body = Gg.subgraph(body).copy()
    body_points = list(body)
    
    # branch index untuk allow revisit (HANYA di percabangan)
    branch_nodes_idx = extract_branch_points(G_body, body_points)
    
    # junction dari huruf kanan dan ke huruf kiri
    ent = entry_node.get(gi, None)  # junction dari huruf kanan
    ext = exit_node.get(gi, None)   # junction ke huruf kiri
    
    # snap ent/ext ke node body terdekat kalau ent/ext tidak ada di body
    if ent is not None and ent not in body:
        ent = min(body_points, key=lambda p: euclidean(p, ent))
    if ext is not None and ext not in body:
        ext = min(body_points, key=lambda p: euclidean(p, ext))
    
    # ambil endpoint (degree==1)
    endpoints = [n for n in G_body.nodes if G_body.degree[n] == 1]
    
    # pakai endpoints kalau ada, kalau tidak pakai semua titik body
    cands = endpoints if endpoints else body_points
    
    start_point = pick_right_top(cands)
    end_point = pick_left_bottom(cands)
    
    # simpan untuk plot label S/E
    start_points[gi] = start_point
    end_points[gi] = end_point
    
    # Jalankan greedy dengan END dipaksa
    body_path = tsp_greedy_with_revisit(
        body_points,
        branch_nodes_idx,
        max_visits=2,
        start_point=start_point,
        end_point=end_point
    )
    
    # ✅ Jahit supaya jalurnya mengikuti skeleton (tidak lompat)
    body_path = densify_path_follow_graph(G_body, body_path)
    
    # Pastikan body_path dimulai dari start_point dan berakhir di end_point
    if len(body_path) > 0:
        if euclidean(body_path[0], start_point) > euclidean(body_path[-1], start_point):
            body_path = list(reversed(body_path))
        if euclidean(body_path[-1], end_point) > euclidean(body_path[0], end_point):
            body_path = list(reversed(body_path))
            
            
        # ============================================================
    # TAMBAHAN: PROSES TSP UNTUK SETIAP DOT/DIAKRITIK
    # ============================================================
    
    for dot_idx, dset in enumerate(dots):
        # Buat graph untuk dot ini
        G_dot = Gg.subgraph(dset).copy()
        dot_points_single = list(dset)
        
        if len(dot_points_single) == 0:
            continue
        
        # Deteksi endpoint dot
        endpoints_dot = [n for n in G_dot.nodes if G_dot.degree[n] == 1]
        
        # Pilih kandidat
        if endpoints_dot:
            cands_dot = endpoints_dot
        else:
            cands_dot = dot_points_single
        
        # Pilih start/end untuk dot
        start_dot = pick_right_top(cands_dot)
        end_dot = pick_left_bottom(cands_dot)
        
        # TSP untuk dot (tanpa revisit)
        dot_path = tsp_greedy_with_revisit(
            dot_points_single,
            [],  # tidak ada branch untuk titik kecil
            max_visits=1,
            start_point=start_dot,
            end_point=end_dot
        )
        
        # Densify path dot
        dot_path = densify_path_follow_graph(G_dot, dot_path)
        
        # Update visit counts
        for pt in dot_path:
            visit_counts[pt] += 1
        
        # SIMPAN dot sebagai subpath terpisah (untuk visualisasi TSP)
        all_subpaths.append(dot_path)
        
        # Hitung chain code untuk dot
        chain_code_dot = []
        for i in range(len(dot_path) - 1):
            p1 = dot_path[i]
            p2 = dot_path[i + 1]
            code = direction_code(p1, p2)
            if code != -1:
                chain_code_dot.append(code)
        
        print(f"\n[DOT {dot_idx+1} untuk huruf {gi+1}] TSP Path: {len(dot_path)} points")
        print(f"[DOT {dot_idx+1}] Chain Code: {chain_code_dot}")        
    
    # gabungkan dot untuk CROPPING (bukan untuk mempengaruhi start/end)
    dot_points = []
    for d in dots:
        dot_points.extend(sorted(list(d), key=lambda p: (p[1], p[0])))
    
    full_path = body_path + dot_points
    
    # update visit + branch nodes
    for pt in body_path:
        visit_counts[pt] += 1
    
    for pt in G_body.nodes:
        if G_body.degree[pt] >= 3:
            branch_nodes.add(pt)
    
    # simpan semua data
    all_subpaths.append(body_path)          # untuk gambar TSP (path body aja)
    all_letters_nodes.append(gset)          # untuk CROPPING per huruf (pakai node huruf)
    all_letter_points.append(set(full_path))  # TAMBAHKAN BARIS INI - untuk cropping
    all_dots.append(dots)                   # untuk simpan dots terpisah

print("JUMLAH HURUF TERDETEKSI:", len(all_subpaths))

        

# ====================
# VISUALISASI SEMUA JALUR
# ====================
visualize_all_paths(combined_skeleton, all_subpaths, visit_counts, branch_nodes)

    
    
# ========================
# SIMPAN SUB-PATH
# ========================
print("\nSub-path details:")

global_counter = 1
for idx, path in enumerate(all_subpaths):
    # Cek apakah sub-path ini mengandung titik dari loop
    is_loop = any(pt in branch_nodes for pt in path)
    
    # Hitung panjang total
    total_length = 0
    for i in range(1, len(path)):
        total_length += euclidean(path[i], path[i - 1])
    
    print(f"\nSub-path {idx+1}: (loop: {is_loop}, panjang: {total_length:.2f})")
    for i in range(len(path)):
        pt = tuple(int(v) for v in path[i])
        if i == 0:
            dist = 0.00
        else:
            dist = euclidean(path[i], path[i - 1])
        print(f"{global_counter}. Point {pt} | Jarak: {dist:.2f}")
        global_counter += 1


# ========================
# SIMPAN FREEMAN CHAIN CODE
# ========================
# Proses Freeman Chain Code untuk setiap subpath
print("\nFreeman Chain Code untuk setiap sub-path:")
for idx, path in enumerate(all_subpaths):
    chain_code = []
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        code = direction_code(p1, p2)
        
        if code != -1:
            chain_code.append(code)
        else:
            # Jika bukan tetangga langsung, lakukan pemecahan langkah dengan batas langkah
            x1, y1 = p1
            x2, y2 = p2
            steps = 0
            max_steps = 1000  # untuk menghindari infinite loop
            while (x1, y1) != (x2, y2) and steps < max_steps:
                dx = np.sign(x2 - x1)
                dy = np.sign(y2 - y1)
                next_x = x1 + dx
                next_y = y1 + dy
                code = direction_code((x1, y1), (next_x, next_y))
                if code != -1:
                    chain_code.append(code)
                    x1, y1 = next_x, next_y
                else:
                    print(f"Gagal menentukan arah dari ({x1}, {y1}) ke ({next_x}, {next_y})")
                    break
                steps += 1
            if steps >= max_steps:
                print(f"⚠  Langkah melebihi batas pada sub-path {idx + 1}, antara titik {p1} dan {p2}")
    
    print(f"Sub-path {idx + 1}:", chain_code)
    # print("Chain Code:", chain_code)












# Buat folder output jika belum ada
import os

output_folder = 'output_potongan_huruf'
os.makedirs(output_folder, exist_ok=True)

huruf_terpotong = []

for idx, pts in enumerate(all_letter_points):
    pts = list(pts)
    if not pts:
        continue

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    min_x = max(min(xs) - 2, 0)
    max_x = min(max(xs) + 3, combined_skeleton.shape[1])
    min_y = max(min(ys) - 2, 0)
    max_y = min(max(ys) + 3, combined_skeleton.shape[0])

    # bikin mask huruf ini saja (biar crop tidak ikut huruf lain walau bbox overlap)
    mask = np.zeros_like(combined_skeleton, dtype=np.uint8)
    for (x, y) in pts:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            mask[y, x] = 1

    potongan = mask[min_y:max_y, min_x:max_x]
    huruf_terpotong.append(potongan)

    filename = os.path.join(output_folder, f"huruf_{idx+1:02}.png")
    cv.imwrite(filename, (potongan * 255).astype(np.uint8))
    print(f"Huruf {idx+1} disimpan: {filename}")

# ============================
# ============================
# VISUALISASI HASIL POTONGAN (KANAN -> KIRI)
# ============================
if huruf_terpotong:
    n = len(huruf_terpotong)
    plt.figure(figsize=(4 * n, 6))

    # urutan tampil: Potongan 1 di kanan
    for i in range(n):
        img = huruf_terpotong[i]

        # posisi subplot dibalik: i=0 (Potongan 1) ditempatkan di kolom paling kanan
        pos = n - i
        plt.subplot(1, n, pos)

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Potongan {i+1}")

    plt.suptitle(f"Pemotongan Huruf Berdasarkan Huruf Terdeteksi\nTotal: {n}", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Tidak ada huruf terpotong ditemukan!")



# ============================
# LABEL OTOMATIS
# ============================

label_global_path = os.path.join(output_folder, "all_labels.txt")
with open(label_global_path, 'w', encoding='utf-8') as f:
    f.write("")

for idx, path in enumerate(all_subpaths):
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    if not xs or not ys:
        continue

    min_x = max(min(xs) - 2, 0)
    max_x = min(max(xs) + 3, combined_skeleton.shape[1])
    min_y = max(min(ys) - 2, 0)
    max_y = min(max(ys) + 3, combined_skeleton.shape[0])

    # Nama file
    nama_gambar = f"huruf_{idx+1:02}.png"
    nama_label = f"huruf_{idx+1:02}.txt"

    # Simpan file label individu
    label_path = os.path.join(output_folder, nama_label)
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write("?")  # placeholder label

    # Tambahkan ke global file
    with open(label_global_path, 'a', encoding='utf-8') as f:
        f.write(f"{nama_gambar} : ?  # bbox: ({min_x}, {min_y}) - ({max_x}, {max_y})\n")







# # ==================== Evaluasi Manual Segmentasi Huruf Pitri ====================

# # Input manual dari pengguna
# jumlah_GT = int(input("Masukkan jumlah Ground Truth (GT): "))
# jumlah_DT = int(input("Masukkan jumlah Deteksi (DT): "))
# TP = int(input("Masukkan jumlah True Positives (TP): "))
# FP = int(input("Masukkan jumlah False Positives (FP): "))
# FN = int(input("Masukkan jumlah False Negatives (FN): "))

# # Validasi konsistensi data (opsional tapi disarankan)
# if TP > jumlah_GT or TP > jumlah_DT:
#     print("\n[PERINGATAN] Nilai TP melebihi GT atau DT, cek kembali input Anda!")

# # Hitung metrik evaluasi
# accuracy = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
# precision = TP / (TP + FP) if (TP + FP) != 0 else 0
# recall = TP / (TP + FN) if (TP + FN) != 0 else 0
# f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

# # Tampilkan hasil evaluasi
# print("\n===== Hasil Evaluasi Otomatis Segmentasi Huruf =====")
# print(f"GT (Ground Truth) : {jumlah_GT}")
# print(f"DT (Detected)     : {jumlah_DT}")
# print(f"TP (Benar)        : {TP}")
# print(f"FP (Salah)        : {FP}")
# print(f"FN (Terlewat)     : {FN}")
# print(f"Accuracy          : {accuracy:.3f}")
# print(f"Precision         : {precision:.3f}")
# print(f"Recall            : {recall:.3f}")
# print(f"F1 Score          : {f1_score:.3f}")


# EVALUASI MANUAL
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# def manual_segmentation_evaluation():
#     print("=== Evaluasi Manual Segmentasi Huruf ===")

#     # Input manual
#     true_count = int(input("Masukkan jumlah huruf sebenarnya (GT): "))
#     predicted_count = int(input("Masukkan jumlah hasil segmentasi/deteksi (DT): "))
#     # TP = int(input("Masukkan jumlah True Positive (TP): "))
#     # FP = int(input("Masukkan jumlah False Positive (FP): "))
#     # FN = int(input("Masukkan jumlah False Negative (FN): "))

#     # TP FP FN YANG OTOMATIS
#     TP = min(predicted_count, true_count)
#     FP = max(predicted_count - true_count, 0)
#     FN = max(true_count - predicted_count, 0)

#     # Hitung metrik evaluasi
#     precision = TP / (TP + FP + 1e-6)
#     recall = TP / (TP + FN + 1e-6)
#     f1 = 2 * precision * recall / (precision + recall + 1e-6)
#     accuracy = TP / (true_count + 1e-6)

#     # Output hasil
#     print("\n=== Hasil Evaluasi ===")
#     print(f"Ground Truth (GT)     : {true_count}")
#     print(f"Detected (DT)         : {predicted_count}")
#     print(f"True Positive (TP)    : {TP}")
#     print(f"False Positive (FP)   : {FP}")
#     print(f"False Negative (FN)   : {FN}")
#     print(f"Precision             : {precision:.3f}")
#     print(f"Recall                : {recall:.3f}")
#     print(f"F1 Score              : {f1:.3f}")
#     print(f"Accuracy              : {accuracy:.3f}")

# # Jalankan fungsi:
# manual_segmentation_evaluation()


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def manual_segmentation_evaluation():
    print("=== Evaluasi Manual Segmentasi Huruf ===")

    # Input manual
    true_count = int(input("Masukkan jumlah huruf sebenarnya (GT): "))
    predicted_count = int(input("Masukkan jumlah hasil segmentasi/deteksi (DT): "))

    # Hitung otomatis TP, FP, FN (pendekatan sederhana)
    TP = min(predicted_count, true_count)
    FP = max(predicted_count - true_count, 0)
    FN = max(true_count - predicted_count, 0)

    # Tambahkan epsilon kecil untuk hindari pembagian 0
    eps = 1e-6

    # Hitung metrik evaluasi
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = TP / (true_count + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)

    # Output hasil
    print("\n=== Hasil Evaluasi ===")
    print(f"Ground Truth (GT)           : {true_count}")
    print(f"Detected (DT)               : {predicted_count}")
    print(f"True Positive (TP)          : {TP}")
    print(f"False Positive (FP)         : {FP}")
    print(f"False Negative (FN)         : {FN}")
    print(f"Precision                   : {precision:.3f}")
    print(f"Recall                      : {recall:.3f}")
    print(f"F1 Score                    : {f1:.3f}")
    print(f"Accuracy                    : {accuracy:.3f}")
    print(f"Intersection over Union (IoU): {iou:.3f}")
    print(f"Dice Coefficient             : {dice:.3f}")

# Jalankan fungsi
manual_segmentation_evaluation()
