"""
FreeCOS-style FDA Synthetic Image Generator
============================================
Paper-exact re-implementation of the FFS (Fractal-FDA Synthesis) module
from: FreeCOS (ICCV 2023) — https://arxiv.org/abs/2307.07245

Pipeline (matches Datasetloader/XCAD_liot.py :: load_frame_fakevessel_gaussian):
  1. Generate a synthetic line image via a parametric stochastic L-system
     (make_fakevessel.py parameters, turtle-free via cv2.line).
  2. FDA: transfer low-frequency amplitude of the real background into the
     synthetic image (source=synthetic, target=real, L=0.3).
  3. Post-process: GaussianBlur(13,13) + uniform(-5,5) noise.

Output:
  Data/FDA_dataset/images/   – FDA-blended composite images
  Data/FDA_dataset/masks/    – binary masks  (0 = background, 255 = crack)

Usage:
  python FreeCOS_FDA_generator.py
"""

import glob
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────
SAMPLES_DIR  = "./samples"       # real unlabeled images
OUTPUT_DIR   = "Data/FDA_dataset"
NUM_IMAGES   = 300              # total composites to generate
FDA_L        = 0.01               # paper value from load_frame_fakevessel_gaussian
# ──────────────────────────────────────────────────────────────────────────


# ── FDA — exact paper implementation (utils/FDA.py + XCAD_liot.py) ────────

def _low_freq_mutate_np(amp_src, amp_trg, L):
    """
    Paper-exact low-frequency swap.
    fftshift centres the spectrum, swaps a (2b+1)×(2b+1) central square,
    then ifftshift restores the layout.
    amp_src / amp_trg: float32 arrays of shape (C, H, W).
    """
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
    _, h, w = a_src.shape
    b   = int(np.floor(np.amin((h, w)) * L))
    c_h = int(np.floor(h / 2.0))
    c_w = int(np.floor(w / 2.0))
    h1, h2 = c_h - b, c_h + b + 1
    w1, w2 = c_w - b, c_w + b + 1
    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def fda_source_to_target_np(src_img, trg_img, L=FDA_L):
    """
    FDA: transfer low-frequency amplitude of trg into src.
    src_img, trg_img: float32 (C, H, W), values in [0, 255].
    Returns float32 (C, H, W).
    """
    fft_src     = np.fft.fft2(src_img, axes=(-2, -1))
    fft_trg     = np.fft.fft2(trg_img, axes=(-2, -1))
    amp_src     = np.abs(fft_src)
    pha_src     = np.angle(fft_src)
    amp_trg     = np.abs(fft_trg)
    amp_src_new = _low_freq_mutate_np(amp_src.copy(), amp_trg, L)
    fft_new     = amp_src_new * np.exp(1j * pha_src)
    result      = np.real(np.fft.ifft2(fft_new, axes=(-2, -1)))
    return result


# ── Fractal L-system — exact parameters from make_fakevessel.py ───────────

# Four rule sets from make_fakevessel.py (line-for-line)
_RULES   = {"F": "F-F[+F-F][-F+F]"}
_RULES_2 = {"F": "F+F[+F[+F]-F]-F+F[+F-F[+F]-F]"}
_RULES_3 = {"F": "F[+F]+F[+F]+F[+F]"}
_RULES_4 = {"F": "F-F-F[+F-F][-F-F]F+F"}


class FractalLSystem:
    """
    Turtle-free re-implementation of LSystem_vessel from make_fakevessel.py.

    Parameter ranges are taken verbatim from the generation loop:
      Width        = (2, 12)          — np.random.randint(Width[0], Width[1]+1)
      Length_range = (90, 150) px     — on paper's 512×512 base; scaled here
      Ratio_LW     = (0.7, 1.0)       — both lam1 and lam2
      Dtheta       = (20, 120)        — stored but unused in draw (see below)
      iteration    = randint(1, 3)    — i.e. 1 or 2  (np excludes upper)
      Axiom        = "[+F-F][-F]" or "F"  (50 / 50)
      Rule pair    = (rules,rules_2) or (rules_4,rules_3)  (50 / 50)

    Angle per '+'/'-' symbol: np.random.randint(1, 5) = 1–4 degrees.
    (In the paper, dtheta_1/dtheta_2 are stored but the draw() method uses
    randint(1,5) directly — the stored dtheta values are never referenced.)
    """

    def __init__(self, img_h, img_w):
        self.img_h = img_h
        self.img_w = img_w

        # Width: (2, 12) inclusive — np.random.randint(2, 13)
        self.width = int(np.random.randint(2, 13))

        # Length: 90–150 px on a 512×512 canvas, scaled to our image size
        scale        = min(img_h, img_w) / 512.0
        self.length  = float(np.random.randint(90, 151)) * scale

        # Iterations: 1 or 2  (np.random.randint(1,3) excludes 3)
        self.iter = int(np.random.randint(1, 3))

        # Both lam values from Ratio_LW = (0.7, 1.0)
        self.lam1 = float(np.random.uniform(0.7, 1.0))
        self.lam2 = float(np.random.uniform(0.7, 1.0))

        # Axiom: 50 % "[+F-F][-F]", 50 % "F"
        self.axiom    = "[+F-F][-F]" if random.random() > 0.5 else "F"
        self.sentence = self.axiom

        # Rule pair selection (50 / 50)
        if random.random() > 0.5:
            self.r1, self.r2 = _RULES,   _RULES_2   # p_vessel > 0.5 branch
        else:
            self.r1, self.r2 = _RULES_4, _RULES_3   # p_vessel ≤ 0.5 branch

    def generate(self):
        """
        Stochastic L-system expansion — exact logic from LSystem_vessel.generate():
          p > 0.4  (60 %) → r1
          p < 0.4  (40 %) → r2
          (r3 exists in paper but is unreachable due to elif condition)
        """
        for _ in range(self.iter):
            new_str = ""
            for ch in self.sentence:
                mapped = ch
                try:
                    p = random.random()
                    if p > 0.4:
                        mapped = self.r1[ch]
                    else:
                        mapped = self.r2[ch]
                except KeyError:
                    pass
                new_str += mapped
            self.sentence = new_str

    def render(self):
        """
        Render the sentence to a binary mask via cv2.line.

        Angle per '+'/'-': np.random.randint(1, 5)  (= 1, 2, 3, or 4 degrees)
        — matches paper's draw() which uses randint(low=1, high=5) per symbol.

        Length per 'F': length*lam1 when inside a branch ('[…]'),
                        length*lam2 on the main stem — mirrors paper's flag logic.
        """
        canvas = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        x     = float(np.random.randint(self.img_w // 4, 3 * self.img_w // 4))
        y     = float(np.random.randint(self.img_h // 4, 3 * self.img_h // 4))
        theta = float(np.random.uniform(0.0, 360.0))
        width = float(self.width)
        stack = []
        flag  = False  # True after '[', False after ']'

        for ch in self.sentence:
            if ch in ('F', 'G'):
                seg_len = self.length * (self.lam1 if flag else self.lam2)
                rad = np.radians(theta)
                nx  = x + seg_len * np.cos(rad)
                ny  = y + seg_len * np.sin(rad)
                p1  = (int(np.clip(x,  0, self.img_w - 1)),
                       int(np.clip(y,  0, self.img_h - 1)))
                p2  = (int(np.clip(nx, 0, self.img_w - 1)),
                       int(np.clip(ny, 0, self.img_h - 1)))
                cv2.line(canvas, p1, p2, 255, max(1, int(round(width))))
                x, y = nx, ny
            elif ch == '+':
                dtheta = float(np.random.randint(1, 5))   # 1–4 deg (paper)
                theta += dtheta
                width  = max(1.0, width * self.lam1)
            elif ch == '-':
                dtheta = float(np.random.randint(1, 5))   # 1–4 deg (paper)
                theta -= dtheta
                width  = max(1.0, width * self.lam2)
            elif ch == '[':
                stack.append((x, y, theta, width))
                flag = True
            elif ch == ']':
                if stack:
                    x, y, theta, width = stack.pop()
                flag = False

        return canvas


def make_fractal_mask(img_h, img_w):
    """Generate one fractal L-system binary mask (0/255) at the given size."""
    ls = FractalLSystem(img_h, img_w)
    ls.generate()
    return ls.render()


# ── Composite builder — matches load_frame_fakevessel_gaussian ────────────

def generate_fda_composite(real_img, crack_mask, L=FDA_L):
    """
    Paper-exact composite construction (load_frame_fakevessel_gaussian):

      1. Build a synthetic line image: lines drawn at random gray intensity
         on a light background (replaces turtle's random-colour makecolor()).
      2. FDA(source=synthetic, target=real, L=0.3) — pulls synthetic into
         the real image's low-frequency style.
      3. GaussianBlur((13,13), sigma=0) + uniform(-5, 5) noise.

    Returns:
      composite – uint8 grayscale
      mask_out  – uint8 binary mask (0 / 255)
    """
    h, w = real_img.shape

    # Step 1 — synthetic line image (bright lines on dark background).
    # Using dark background + bright lines ensures lines remain visible after
    # FDA maps the image toward the (typically dark) real X-ray target domain.
    real_mean = float(real_img.mean())
    bg_val    = max(0.0, real_mean - 20.0)          # slightly below real mean
    offset    = float(np.random.randint(100, 201))           # random offset: 100–200
    line_val  = min(255.0, real_mean + offset)           # well above real mean
    synth = np.full((h, w), bg_val, dtype=np.float32)
    synth[crack_mask > 0] = line_val

    # Expand to (1, H, W) — paper processes (C, H, W) arrays
    im_src = synth[np.newaxis, :, :]                            # (1,H,W)
    im_trg = real_img.astype(np.float32)[np.newaxis, :, :]      # (1,H,W)

    # Step 2 — FDA
    src_in_trg = fda_source_to_target_np(im_src, im_trg, L=L)
    composite  = np.squeeze(np.clip(src_in_trg, 0.0, 255.0), axis=0)  # (H,W)

    # Step 3 — GaussianBlur + noise (exact paper post-processing)
    composite = cv2.GaussianBlur(composite.astype(np.float32), (13, 13), 0)
    noise     = np.random.uniform(-5.0, 5.0, composite.shape)
    composite = np.clip(composite + noise, 0.0, 255.0).astype(np.uint8)

    mask_out = (crack_mask > 0).astype(np.uint8) * 255
    return composite, mask_out


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/masks",  exist_ok=True)

    sample_paths = sorted(
        glob.glob(os.path.join(SAMPLES_DIR, "*.png")) +
        glob.glob(os.path.join(SAMPLES_DIR, "*.jpg")) +
        glob.glob(os.path.join(SAMPLES_DIR, "*.tif"))
    )
    if not sample_paths:
        raise FileNotFoundError(
            f"No images found in {SAMPLES_DIR!r}. "
            "Update SAMPLES_DIR to your real samples folder.")

    print(f"\n── FreeCOS FDA generator (paper-exact) ──")
    print(f"   Real samples : {len(sample_paths)} images  ({SAMPLES_DIR})")
    print(f"   Target count : {NUM_IMAGES} composites")
    print(f"   FDA L        : {FDA_L}  (load_frame_fakevessel_gaussian)")
    print(f"   Output dir   : {OUTPUT_DIR}\n")

    for i in tqdm(range(NUM_IMAGES), desc="Generating FDA composites"):
        real_path = random.choice(sample_paths)
        real      = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        if real is None:
            continue
        h, w = real.shape

        crack_mask = make_fractal_mask(h, w)
        comp, mask_out = generate_fda_composite(real, crack_mask)

        cv2.imwrite(f"{OUTPUT_DIR}/images/fda_{i:04d}.png", comp)
        cv2.imwrite(f"{OUTPUT_DIR}/masks/fda_{i:04d}.png",  mask_out)

    print(f"\nDone — {NUM_IMAGES} pairs saved to {OUTPUT_DIR}/")
