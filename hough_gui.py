"""
hough_refined_both.py

Refined implementations of:
 - slope-intercept Hough (m, b)
 - rho-theta Hough (rho, theta)

GUI inputs:
 - Load image
 - Edge threshold (0..1) used to adapt Canny thresholds
 - Top-K lines to detect

Fixed internal Hough parameters (hard-coded):
 - SI m range: (-5, 5, 400)
 - SI b steps: 400
 - RT rho_res = 1, theta_steps = 180
 - Peak votes threshold = 40
"""
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --------------------------
# Fixed Hough parameters
# --------------------------
SI_M_MIN, SI_M_MAX, SI_M_STEPS = -5.0, 5.0, 400
SI_B_STEPS = 400
RT_RHO_RES = 1
RT_THETA_STEPS = 180
PEAK_ABS_THRESHOLD = 40

# --------------------------
# Utility helpers
# --------------------------
def pil_open_and_resize(path_or_pil, maxdim=900):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert('RGB')
    else:
        img = path_or_pil.convert('RGB')
    if max(img.size) > maxdim:
        img = ImageOps.contain(img, (maxdim, maxdim))
    return img

def rgb2gray_np(arr):
    if arr.ndim == 2:
        return arr.astype(np.float32)
    return (0.2989*arr[...,0] + 0.5870*arr[...,1] + 0.1140*arr[...,2]).astype(np.float32)

def draw_lines_pil(img_pil, lines, mode='rho_theta', color=(255,0,0), width=2):
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    if mode == 'rho_theta':
        for rho, theta in lines:
            a = np.cos(theta); b = np.sin(theta)
            x0 = a*rho; y0 = b*rho
            x1 = int(x0 + 2000*(-b)); y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b)); y2 = int(y0 - 2000*(a))
            draw.line([(x1,y1),(x2,y2)], fill=color, width=width)
    else: # slope-intercept
        for m,b in lines:
            if abs(m) < 1e-6:
                y = int(round(b))
                draw.line([(0,y),(w,y)], fill=color, width=width)
            else:
                x0, x1 = 0, w
                y0, y1 = m*x0 + b, m*x1 + b
                draw.line([(x0,y0),(x1,y1)], fill=color, width=width)
    return img_pil

# --------------------------
# Peak detection (NMS + centroid refinement)
# --------------------------
def detect_peaks(acc, topk=10, nsize=11, threshold_abs=PEAK_ABS_THRESHOLD, refine=True):
    accf = acc.astype(float)
    peaks = []
    half = nsize//2
    for _ in range(topk):
        idx = np.unravel_index(np.argmax(accf), accf.shape)
        val = accf[idx]
        if val < threshold_abs:
            break
        r, c = idx
        # local window bounds
        r0, r1 = max(0, r-half), min(acc.shape[0], r+half+1)
        c0, c1 = max(0, c-half), min(acc.shape[1], c+half+1)
        window = acc[r0:r1, c0:c1].astype(float)
        if refine:
            # centroid refinement within window for sub-bin estimate
            Y, X = np.indices(window.shape)
            total = window.sum()
            if total > 0:
                yc = (Y * window).sum() / total + r0
                xc = (X * window).sum() / total + c0
            else:
                yc, xc = r, c
        else:
            yc, xc = r, c
        peaks.append(((yc, xc), int(val)))
        # suppress neighborhood
        accf[r0:r1, c0:c1] = 0.0
    return peaks

# --------------------------
# Hough: Rho-Theta (vectorized)
# --------------------------
def hough_rho_theta(edges, rho_res=RT_RHO_RES, theta_steps=RT_THETA_STEPS):
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return None, None, None
    h, w = edges.shape
    thetas = np.deg2rad(np.linspace(0.0, 180.0, theta_steps, endpoint=False))
    diag = int(np.ceil(np.hypot(h, w)))
    rhos = np.linspace(-diag, diag, int(2*diag/rho_res)+1)
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Vectorized loop per point computing r for all thetas and incrementing
    # We do this in chunks to limit memory usage if many edge points
    pts = list(zip(ys, xs))
    CHUNK = 2048  # tune: larger is faster but uses more memory
    for i in range(0, len(pts), CHUNK):
        chunk = pts[i:i+CHUNK]
        ys_c = np.array([p[0] for p in chunk])[:, None]  # (N,1)
        xs_c = np.array([p[1] for p in chunk])[:, None]  # (N,1)
        # compute (N, T) matrix of rho values
        rvals = xs_c * cos_t[None, :] + ys_c * sin_t[None, :]
        # convert to indices
        idx = np.round((rvals - rhos[0]) / rho_res).astype(int)
        # mask valid
        valid = (idx >= 0) & (idx < len(rhos))
        # flatten indices and use np.add.at on accumulator
        # map 2D indices (rho_idx, theta_idx) -> linear pairs for add.at
        rho_idx_flat = idx[valid]
        theta_idx_flat = np.nonzero(valid)[1]  # careful: returns theta indices repeated
        # but we need corresponding theta indices per valid element; above is wrong approach
        # Better approach: iterate per theta for this chunk (vectorize over points)
        for t_i in range(len(thetas)):
            col_idx = idx[:, t_i]
            valid_col = (col_idx >= 0) & (col_idx < len(rhos))
            if not np.any(valid_col):
                continue
            rows = col_idx[valid_col]
            # np.add.at to accumulate counts on (rho,theta) slice
            np.add.at(acc[:, t_i], rows, 1)
    return rhos, thetas, acc

# --------------------------
# Hough: Slope-Intercept (vectorized per slope)
# --------------------------
def hough_slope_intercept(edges, m_min=SI_M_MIN, m_max=SI_M_MAX, m_steps=SI_M_STEPS, b_steps=SI_B_STEPS):
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return None, None, None
    ms = np.linspace(m_min, m_max, int(m_steps))
    # compute b for all (m,x,y) via loop over ms to limit memory usage
    # build b range from all points across sampled ms
    # first make coarse estimate of b_min/max using only a subset for speed
    subset = slice(0, len(xs), max(1, len(xs)//2000))
    bs_all = []
    for m in ms:
        bs_all.extend((ys[subset] - m*xs[subset]).tolist())
    bs_all = np.array(bs_all)
    b_min, b_max = bs_all.min(), bs_all.max()
    bs = np.linspace(b_min, b_max, int(b_steps))
    acc = np.zeros((len(ms), len(bs)), dtype=np.int32)

    # Now vote: for each slope (vectorized over points)
    for mi, m in enumerate(ms):
        b_vals = ys - m * xs  # vector length = number of edge points
        # get bin indices for b
        b_idx = np.searchsorted(bs, b_vals)
        valid = (b_idx >= 0) & (b_idx < len(bs))
        if not np.any(valid):
            continue
        # increment accumulator row mi at columns b_idx[valid]
        np.add.at(acc[mi], b_idx[valid], 1)
    return ms, bs, acc

# --------------------------
# GUI + Pipeline
# --------------------------
class HoughApp:
    def __init__(self, root):
        self.root = root
        root.title("Hough (refined) - both methods")
        self.img_pil = None

        frm = tk.Frame(root); frm.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        tk.Button(frm, text="Load Image", command=self.load_image).pack(fill='x', pady=4)
        tk.Button(frm, text="Use Sample Image", command=self.use_sample).pack(fill='x', pady=4)
        tk.Label(frm, text="Edge threshold (0..1) [fraction]").pack(anchor='w')
        self.ent_thresh = tk.Entry(frm); self.ent_thresh.insert(0, "0.33"); self.ent_thresh.pack(fill='x')
        tk.Label(frm, text="Top-K lines").pack(anchor='w')
        self.ent_k = tk.Entry(frm); self.ent_k.insert(0, "6"); self.ent_k.pack(fill='x')
        tk.Button(frm, text="Run", command=self.run).pack(fill='x', pady=8)
        tk.Button(frm, text="Quit", command=root.quit).pack(fill='x', pady=4)

        self.fig, self.axes = plt.subplots(2,3, figsize=(12,7))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill='both', expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),("All","*.*")])
        if not path:
            return
        self.img_pil = pil_open_and_resize(path)
        self.show_preview()

    def use_sample(self):
        # use a small synthetic test if user wants
        img = Image.new('RGB', (500,500), (255,255,255))
        d = ImageDraw.Draw(img)
        d.line([(50,450),(450,50)], fill=(0,0,0), width=4)
        d.line([(0,250),(500,250)], fill=(0,0,0), width=3)
        d.line([(250,0),(250,500)], fill=(0,0,0), width=3)
        self.img_pil = img
        self.show_preview()

    def show_preview(self):
        for ax in self.fig.axes:
            ax.clear(); ax.axis('off')
        if self.img_pil is None:
            self.fig.axes[0].text(0.5,0.5,"No image loaded", ha='center')
        else:
            arr = np.array(self.img_pil)
            gray = rgb2gray_np(arr)
            self.fig.axes[0].imshow(arr.astype(np.uint8)); self.fig.axes[0].set_title("Original")
            self.fig.axes[1].imshow(gray, cmap='gray'); self.fig.axes[1].set_title("Grayscale")
        self.canvas.draw()

    def run(self):
        if self.img_pil is None:
            messagebox.showerror("No image", "Load an image first."); return
        try:
            thr = float(self.ent_thresh.get())
            if not (0.0 <= thr <= 1.0):
                raise ValueError("threshold must be in [0,1]")
            topk = int(self.ent_k.get())
        except Exception as e:
            messagebox.showerror("Parameter error", str(e)); return

        arr = np.array(self.img_pil)
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # blur to reduce noise
        gray_blur = cv2.GaussianBlur(gray, (5,5), 1.0)

        # Canny thresholds using median heuristic scaled by thr
        med = np.median(gray_blur)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * med) * thr)
        upper = int(min(255, (1.0 + sigma) * med))
        if lower >= upper:
            lower = int(0.5 * upper)
        edges = cv2.Canny(gray_blur, lower, upper, apertureSize=3, L2gradient=True)
        edges_bin = (edges > 0).astype(np.uint8)

        # Slope-Intercept Hough
        ms, bs, acc_si = hough_slope_intercept(edges_bin)
        peaks_si = []
        if acc_si is not None:
            peaks_si = detect_peaks(acc_si, topk, nsize=11, threshold_abs=PEAK_ABS_THRESHOLD, refine=True)

        # Rho-Theta Hough (vectorized)
        rhos, thetas, acc_rt = hough_rho_theta(edges_bin, rho_res=RT_RHO_RES, theta_steps=RT_THETA_STEPS)
        peaks_rt = detect_peaks(acc_rt, topk, nsize=11, threshold_abs=PEAK_ABS_THRESHOLD, refine=True)

        # Map peaks to actual parameters
        lines_si = []
        for (yc, xc), votes in peaks_si:
            # yc is (float) row index ~ m_idx, xc ~ b_idx in SI accumulator
            m_idx = int(round(yc))
            b_idx = int(round(xc))
            m_idx = np.clip(m_idx, 0, len(ms)-1)
            b_idx = np.clip(b_idx, 0, len(bs)-1)
            lines_si.append((ms[m_idx], bs[b_idx]))

        lines_rt = []
        for (yc, xc), votes in peaks_rt:
            rho_idx = int(round(yc))
            theta_idx = int(round(xc))
            rho_idx = np.clip(rho_idx, 0, len(rhos)-1)
            theta_idx = np.clip(theta_idx, 0, len(thetas)-1)
            lines_rt.append((rhos[rho_idx], thetas[theta_idx]))

        # Draw overlays
        pil_si = self.img_pil.copy()
        pil_rt = self.img_pil.copy()
        pil_si = draw_lines_pil(pil_si, lines_si, mode='slope_intercept', color=(255,0,0), width=2)
        pil_rt = draw_lines_pil(pil_rt, lines_rt, mode='rho_theta', color=(0,0,255), width=2)

        # Display everything
        for ax in self.fig.axes:
            ax.clear(); ax.axis('off')
        self.fig.axes[0].imshow(arr.astype(np.uint8)); self.fig.axes[0].set_title("Original")
        self.fig.axes[1].imshow(gray_blur, cmap='gray'); self.fig.axes[1].set_title("Blurred gray")
        self.fig.axes[2].imshow(edges, cmap='gray'); self.fig.axes[2].set_title(f"Edges")
        if acc_si is not None:
            self.fig.axes[3].imshow((acc_si.astype(float)/ (acc_si.max()+1e-6)), cmap='inferno'); self.fig.axes[3].set_title("Accumulator: Slope-Intercept")
        else:
            self.fig.axes[3].text(0.5,0.5,"No SI accumulator", ha='center')
        self.fig.axes[4].imshow((acc_rt.astype(float)/ (acc_rt.max()+1e-6)), cmap='inferno'); self.fig.axes[4].set_title("Accumulator: Rho-Theta")
        self.fig.axes[5].imshow(np.array(pil_rt)); self.fig.axes[5].set_title("RT detected lines (blue)")
        self.canvas.draw()

        # pop-up window showing SI overlay too
        fig2, ax2 = plt.subplots(figsize=(5,6)); ax2.imshow(np.array(pil_si)); ax2.set_title("SI detected lines (red)"); ax2.axis('off'); fig2.show()

        # print summary
        print("Detected Slope-Intercept lines (m, b):")
        for m,b in lines_si:
            print(f"  m={m:.4f}, b={b:.2f}")
        print("\nDetected Rho-Theta lines (rho, theta deg):")
        for rho,theta in lines_rt:
            print(f"  rho={rho:.2f}, theta={np.degrees(theta):.2f} deg")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HoughApp(root)
    root.geometry("1250x800")
    root.mainloop()
