import numpy as np
import cv2
import random
import os
import glob
from tqdm import tqdm
from skimage import exposure
from FreeCOS_FDA_generator import generate_fda_composite

# ============================ CONFIGURATION =============================
# Update values here to tune generation behavior globally.

Config_number = "21"

PATHS = {
    "output_dir": "Data/generalized_dataset",      # generated images/masks
    "samples_dir": "./samples",                    # real images used as FDA target
    "input_dir": "./output_images/",               # real background source for mode='real'
}

RUN_CONFIG = {
    "mode": "simple",                                 # 'simple' | 'real' | 'fda'
    "num_images": 500,
    "image_size": 800,
}

FDA_CONFIG = {
    "L_range": (0.1, 0.9),
    "L_round_digits": 2,
}

MASK_LABELS = {
    "background": 0,
    "line": 255,
    "shape": 125,
}

NOISE_CONFIG = {
    "apply_blur_probability": 0.3,
    "blur_kernel_choices": [3, 5, 7],
    "blur_sigma_range": (1, 2),
    "apply_gaussian_probability": 0.1,
    "apply_poisson_probability": 0.1,
    "apply_salt_pepper_probability": 0.1,
    "apply_speckle_probability": 0.1,
    "gaussian_mean": 0,
    "gaussian_std": 20,
    "salt_pepper_amount": 0.02,
    "salt_vs_pepper": 0.5,
    "speckle_std": 0.2,
}

# ============================

LAYER_ORDER_CONFIG = {
    # True: all objects are interleaved randomly and opacity is forced to 1.0.
    # False: non-fracture objects are drawn first, then fractures are added last.
    "random_layer_order": True,
}

DRAWING_CONFIG = {
    "thickness_range": (1, 5),
    "color_range": (151, 230),
    "bg_color_range": (20, 150),
    "opacity_range": (0.9, 1.0),
    "soft_edge_probability": 0.5,
    "soft_edge_kernels": [3, 5, 7],
    "shape_soft_edge_kernels": [7, 11, 15],
}

CRACK_CONFIG = {
    "start_region_divisor": 4,                      # start in center-ish region
    "main_length_range": (100, 800),
    "color_jitter_range": (0, 0),
    "preferred_angle_range": (0, 2 * np.pi),
    "angle_deviation": 0.2,
    "preferred_angle_drift": (-0.1, 0.1),
    "step_size_range": (1.0, 3.0),
    "thickness_variation_range": (0.3, 2.5),
    "branch_probability": 0.01,
    "branch_after_step": 20,
    "branch_side_choices": [-1, 1],
    "branch_angle_range": (0.3, 0.8),
    "branch_length_range": (20, 100),
    "branch_thickness_drop_range": (1, 2),
    "branch_angle_jitter": (-0.2, 0.2),
    "branch_step_range": (0.8, 2.0),
    "branch_color_jitter": (-12, 12),
    "edge_margin": 1,
}

SHAPE_CONFIG = {
    "smallcircle_radius_range": (1, 6),
    "spot_radius_x_range": (2, 18),
    "spot_radius_ratio_range": (0.4, 1.6),
    "spot_point_count_range": (10, 22),
    "spot_edge_perturbation": (-0.4, 0.4),
    "circle_radius_min": 10,
    "circle_radius_divisor": 6,
    "square_side_range": (1, 4),                    # max side = img_size // divisor
}

LAYER_OBJECT_COUNTS = {
    # You can set a fixed count (e.g., 0, 12) or a range tuple (min, max).
    "fracture": (1, 20),
    "smallcircle": (0, 100),
    "spot": (0, 100),
    "circle": (0, 0),
    "rectangle": (0, 0),
    "square": (0, 0),
    "triangle": (0, 0),
}



# Backward-compatible aliases used throughout the code.
img_size = RUN_CONFIG["image_size"]
output_dir = PATHS["output_dir"]
samples_dir = PATHS["samples_dir"]
input_dir = PATHS["input_dir"]
mode = RUN_CONFIG["mode"]
num_images = RUN_CONFIG["num_images"]
thickness_range = DRAWING_CONFIG["thickness_range"]
color_range = DRAWING_CONFIG["color_range"]
bg_color_range = DRAWING_CONFIG["bg_color_range"]
opacity_range = DRAWING_CONFIG["opacity_range"]
soft_edge_probability = DRAWING_CONFIG["soft_edge_probability"]

BACKGROUND_CLASS = MASK_LABELS["background"]
LINE_CLASS = MASK_LABELS["line"]
SHAPE_CLASS = MASK_LABELS["shape"]

# Create output directories
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/masks",  exist_ok=True)


def blend_with_opacity(base_image, overlay_image, overlay_mask, opacity):
    blended = base_image.copy().astype(np.float32)
    visible_pixels = overlay_mask > 0

    if np.any(visible_pixels):
        blended[visible_pixels] = (
            base_image[visible_pixels].astype(np.float32) * (1.0 - opacity)
            + overlay_image[visible_pixels].astype(np.float32) * opacity
        )

    return np.clip(blended, 0, 255).astype(np.uint8)


def blend_with_soft_edges(base_image, overlay_image, overlay_mask, opacity, blur_kernel):
    alpha = (overlay_mask > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (blur_kernel, blur_kernel), 0)
    alpha = np.clip(alpha * opacity, 0.0, 1.0)

    blended = (
        base_image.astype(np.float32) * (1.0 - alpha)
        + overlay_image.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)

# ===== BACKGROUND GENERATION FUNCTIONS =====

# Generate realistic fracture using improved random walk with branching
def realistic_crack_fracture(size, gray_color):
    """
    Generate realistic crack using improved random walk with branching.
    Creates natural-looking cracks with variable width and branching patterns.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Start from random position
    start_div = CRACK_CONFIG["start_region_divisor"]
    x, y = random.randint(size // start_div, 3 * size // start_div), random.randint(size // start_div, 3 * size // start_div)
    
    # Main crack parameters
    main_length = random.randint(*CRACK_CONFIG["main_length_range"])
    base_thickness = random.randint(*thickness_range)
    color_jitter_range = CRACK_CONFIG["color_jitter_range"]
    
    # Direction bias for more natural cracks
    preferred_angle = random.uniform(*CRACK_CONFIG["preferred_angle_range"])
    angle_deviation = CRACK_CONFIG["angle_deviation"]
    
    points = [(x, y)]
    
    # Generate main crack path
    for i in range(main_length):
        # Add some randomness to angle but bias towards preferred direction
        angle = preferred_angle + random.uniform(-angle_deviation, angle_deviation)
        preferred_angle += random.uniform(*CRACK_CONFIG["preferred_angle_drift"]) 
        
        # Variable step size for more natural appearance
        step_size = random.uniform(*CRACK_CONFIG["step_size_range"])
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_x = int(np.clip(x + dx, 0, size - 1))
        new_y = int(np.clip(y + dy, 0, size - 1))
        
        # Variable thickness along crack
        thickness_variation = random.uniform(*CRACK_CONFIG["thickness_variation_range"])
        current_thickness = max(1, int(base_thickness * thickness_variation))
        segment_color = int(np.clip(gray_color + random.randint(*color_jitter_range), 0, 255))
        
        cv2.line(img, (x, y), (new_x, new_y), segment_color, current_thickness)
        cv2.line(mask, (x, y), (new_x, new_y), LINE_CLASS, current_thickness)
        
        # Add branching with low probability
        if random.random() < CRACK_CONFIG["branch_probability"] and i > CRACK_CONFIG["branch_after_step"]:
            branch_angle = angle + random.choice(CRACK_CONFIG["branch_side_choices"]) * random.uniform(*CRACK_CONFIG["branch_angle_range"])
            branch_length = random.randint(*CRACK_CONFIG["branch_length_range"])
            branch_thickness = max(1, current_thickness - random.randint(*CRACK_CONFIG["branch_thickness_drop_range"]))
            
            bx, by = new_x, new_y
            for _ in range(branch_length):
                branch_angle += random.uniform(*CRACK_CONFIG["branch_angle_jitter"])
                bstep = random.uniform(*CRACK_CONFIG["branch_step_range"])
                bdx = bstep * np.cos(branch_angle)
                bdy = bstep * np.sin(branch_angle)
                
                bnew_x = int(np.clip(bx + bdx, 0, size - 1))
                bnew_y = int(np.clip(by + bdy, 0, size - 1))
                branch_color = int(np.clip(segment_color + random.randint(*CRACK_CONFIG["branch_color_jitter"]), 0, 255))
                
                cv2.line(img, (bx, by), (bnew_x, bnew_y), branch_color, branch_thickness)
                cv2.line(mask, (bx, by), (bnew_x, bnew_y), LINE_CLASS, branch_thickness)
                
                bx, by = bnew_x, bnew_y
                
                # Stop if we hit edge
                edge_margin = CRACK_CONFIG["edge_margin"]
                if bx <= edge_margin or bx >= size - (edge_margin + 1) or by <= edge_margin or by >= size - (edge_margin + 1):
                    break
        
        x, y = new_x, new_y
        points.append((x, y))
        
        # Stop if we hit edge
        edge_margin = CRACK_CONFIG["edge_margin"]
        if x <= edge_margin or x >= size - (edge_margin + 1) or y <= edge_margin or y >= size - (edge_margin + 1):
            break
    
    return img, mask


def draw_shape(image, mask, shape_type, gray, opacity_override=None):
    img_size = image.shape[0]  # Ensure img_size is defined based on input image
    shape_img = np.zeros_like(image, dtype=np.uint8)
    shape_mask = np.zeros_like(mask, dtype=np.uint8)

    if shape_type == 'smallcircle':
        center = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
        radius = random.randint(*SHAPE_CONFIG["smallcircle_radius_range"])
        cv2.circle(shape_img, center, radius, gray, -1)
        cv2.circle(shape_mask, center, radius, SHAPE_CLASS, -1)

    elif shape_type == 'spot':
        # Irregular organic spot: distorted ellipse polygon
        cx = random.randint(0, img_size - 1)
        cy = random.randint(0, img_size - 1)
        rx = random.randint(*SHAPE_CONFIG["spot_radius_x_range"])
        ry = int(rx * random.uniform(*SHAPE_CONFIG["spot_radius_ratio_range"]))
        rot = random.uniform(0, 2 * np.pi)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        num_pts = random.randint(*SHAPE_CONFIG["spot_point_count_range"])
        pts = []
        for k in range(num_pts):
            theta = 2 * np.pi * k / num_pts
            # Radial perturbation for irregular edge
            r = 1.0 + random.uniform(*SHAPE_CONFIG["spot_edge_perturbation"])
            ex = r * rx * np.cos(theta)
            ey = r * ry * np.sin(theta)
            px = int(np.clip(cx + ex * cos_r - ey * sin_r, 0, img_size - 1))
            py = int(np.clip(cy + ex * sin_r + ey * cos_r, 0, img_size - 1))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(shape_img, [pts], gray)
        cv2.fillPoly(shape_mask, [pts], SHAPE_CLASS)


    if shape_type == 'circle':
        center = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
        radius = random.randint(SHAPE_CONFIG["circle_radius_min"], img_size // SHAPE_CONFIG["circle_radius_divisor"])
        cv2.circle(shape_img, center, radius, gray, -1)
        cv2.circle(shape_mask, center, radius, SHAPE_CLASS, -1)

    elif shape_type == 'rectangle':
        pt1 = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
        pt2 = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
        cv2.rectangle(shape_img, pt1, pt2, gray, -1)
        cv2.rectangle(shape_mask, pt1, pt2, SHAPE_CLASS, -1)

    elif shape_type == 'square':
        side = random.randint(SHAPE_CONFIG["square_side_range"][0], img_size // SHAPE_CONFIG["square_side_range"][1])
        x = random.randint(0, img_size - side)
        y = random.randint(0, img_size - side)
        pt1 = (x, y)
        pt2 = (x + side, y + side)
        cv2.rectangle(shape_img, pt1, pt2, gray, -1)
        cv2.rectangle(shape_mask, pt1, pt2, SHAPE_CLASS, -1)

    elif shape_type == 'triangle':
        pts = np.array([
            [random.randint(0, img_size - 1), random.randint(0, img_size - 1)],
            [random.randint(0, img_size - 1), random.randint(0, img_size - 1)],
            [random.randint(0, img_size - 1), random.randint(0, img_size - 1)]
        ], np.int32)
        cv2.fillPoly(shape_img, [pts], gray)
        cv2.fillPoly(shape_mask, [pts], SHAPE_CLASS)

    opacity = 1.0 if opacity_override is not None else random.uniform(*opacity_range)
    if random.random() < soft_edge_probability:
        edge_blur = random.choice(DRAWING_CONFIG["shape_soft_edge_kernels"])
        image = blend_with_soft_edges(image, shape_img, shape_mask, opacity, edge_blur)
    else:
        image = blend_with_opacity(image, shape_img, shape_mask, opacity)

    mask = np.where(shape_mask > 0, shape_mask, mask)

    return image, mask

# Gaussian noise
def add_gaussian_noise(image, mean=0, std=20):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Salt and pepper noise
def add_salt_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    noisy = image.copy()
    num_salt = int(amount * image.size * salt_vs_pepper)
    num_pepper = int(amount * image.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 255
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

# Poisson noise
def add_poisson_noise(image):
    noisy = np.random.poisson(image.astype(np.uint8)).astype(np.uint8)
    return np.clip(noisy, 0, 255)

# Speckle noise
def add_speckle_noise(image, std=0.2):
    noise = np.random.randn(*image.shape) * std
    noisy = image + image * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
    

# LIOT
def liot(image, window_size=3):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the shape of the image
    h, w = image.shape
    pad = window_size // 2

    # Pad the image to handle edges
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    # Prepare the output image
    liot_image = np.zeros_like(image, dtype=np.float32)

    # Apply the LIOT transformation
    for y in range(h):
        for x in range(w):
            # Extract the local window
            window = padded_image[y:y+window_size, x:x+window_size].flatten()
            center = window[len(window) // 2]
            
            # Calculate the relative intensity order
            order = np.sum(window > center) / (window_size**2 - 1)
            liot_image[y, x] = order
    
    # Normalize the result to 0-255
    liot_image = exposure.rescale_intensity(liot_image, out_range=(0, 255)).astype(np.uint8)
    
    return liot_image

def apply_circular_mask(image, mask):
    """
    Apply a circular mask to the image and mask.
    Inside the circle: fully visible
    Outside the circle: 10% visible in image, completely black (0) in mask
    """
    h, w = image.shape[:2]
    center_x = w // 2 + random.randint(-w//8, w//8)  # Slight randomness in center position
    center_y = h // 2 + random.randint(-h//8, h//8)
    
    # Random radius between 35% to 45% of the smaller dimension
    min_dim = min(h, w)
    radius = random.randint(int(min_dim * 0.35), int(min_dim * 0.45))
    
    # Create circular mask
    y_grid, x_grid = np.ogrid[:h, :w]
    distances = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    circle_mask = distances <= radius
    
    # Apply to image: outside is 10% visible (multiply by 0.1)
    image_with_circle = image.copy()
    image_with_circle[~circle_mask] = (image_with_circle[~circle_mask] * 0.1).astype(np.uint8)
    
    # Apply to mask: outside is completely black (0)
    mask_with_circle = mask.copy()
    mask_with_circle[~circle_mask] = 0
    
    return image_with_circle, mask_with_circle


def _resolve_object_count(value):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return random.randint(int(value[0]), int(value[1]))
    return int(value)

def Generate_layers(image, mask):
    random_layer_order = LAYER_ORDER_CONFIG["random_layer_order"]

    object_counts = {}
    for object_name, count_spec in LAYER_OBJECT_COUNTS.items():
        object_counts[object_name] = _resolve_object_count(count_spec)

    # True mode: randomly interleave all objects and force opacity to 1.0.
    if random_layer_order:
        draw_plan = []
        for object_name, count in object_counts.items():
            if count > 0:
                draw_plan.extend([object_name] * count)

        random.shuffle(draw_plan)

        for object_name in draw_plan:
            if object_name == 'fracture':
                gray_color = random.randint(*color_range)
                obj, obj_mask = realistic_crack_fracture(img_size, gray_color)

                opacity = 1.0
                if random.random() < soft_edge_probability:
                    image = blend_with_soft_edges(image, obj, obj_mask, opacity, random.choice(DRAWING_CONFIG["soft_edge_kernels"]))
                else:
                    image = blend_with_opacity(image, obj, obj_mask, opacity)
                mask = np.where(obj_mask > 0, obj_mask, mask)
            else:
                gray = random.randint(0, 255)
                image, mask = draw_shape(image, mask, object_name, gray, opacity_override=1.0)

    # False mode: draw non-fracture objects first, then add fractures last.
    else:
        non_fracture_plan = []
        for object_name, count in object_counts.items():
            if object_name != 'fracture' and count > 0:
                non_fracture_plan.extend([object_name] * count)

        random.shuffle(non_fracture_plan)
        for object_name in non_fracture_plan:
            gray = random.randint(0, 255)
            image, mask = draw_shape(image, mask, object_name, gray)

        for _ in range(max(0, object_counts.get("fracture", 0))):
            gray_color = random.randint(*color_range)
            obj, obj_mask = realistic_crack_fracture(img_size, gray_color)

            opacity = random.uniform(*opacity_range)
            if random.random() < soft_edge_probability:
                image = blend_with_soft_edges(image, obj, obj_mask, opacity, random.choice(DRAWING_CONFIG["soft_edge_kernels"]))
            else:
                image = blend_with_opacity(image, obj, obj_mask, opacity)
            mask = np.where(obj_mask > 0, obj_mask, mask)


    # Optionally apply Gaussian blur
    if random.random() < NOISE_CONFIG["apply_blur_probability"]:
        ksize = random.choice(NOISE_CONFIG["blur_kernel_choices"])
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=random.randint(*NOISE_CONFIG["blur_sigma_range"]))

    if random.random() < NOISE_CONFIG["apply_gaussian_probability"]:
        image = add_gaussian_noise(image, mean=NOISE_CONFIG["gaussian_mean"], std=NOISE_CONFIG["gaussian_std"])
    if random.random() < NOISE_CONFIG["apply_poisson_probability"]:
        image = add_poisson_noise(image)
    if random.random() < NOISE_CONFIG["apply_salt_pepper_probability"]:
        image = add_salt_pepper_noise(image, amount=NOISE_CONFIG["salt_pepper_amount"], salt_vs_pepper=NOISE_CONFIG["salt_vs_pepper"])
    if random.random() < NOISE_CONFIG["apply_speckle_probability"]:
        image = add_speckle_noise(image, std=NOISE_CONFIG["speckle_std"])

    return image, mask

# --- Main Processing Loop ---
# Pre-load real sample images for the FDA pipeline
_sample_paths = sorted(
    glob.glob(os.path.join(samples_dir, "*.png")) +
    glob.glob(os.path.join(samples_dir, "*.jpg")) +
    glob.glob(os.path.join(samples_dir, "*.tif"))
)
_sample_imgs = [
    cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
    for p in _sample_paths
    if cv2.imread(p, cv2.IMREAD_GRAYSCALE) is not None
]

# ── Mode 1: simple background ──────────────────────────────────────────────
if mode == "simple":
    for i in tqdm(range(num_images), desc="Simple background"):
        bg_color = random.randint(*bg_color_range)
        image = np.full((img_size, img_size), bg_color, dtype=np.uint8)
        mask  = np.zeros((img_size, img_size), dtype=np.uint8)
        image, mask = Generate_layers(image, mask)
        cv2.imwrite(f"{output_dir}/images/image_{i:03d}_{Config_number}.png", image)
        cv2.imwrite(f"{output_dir}/masks/mask_{i:03d}_{Config_number}.png",   mask)

# ── Mode 2: real background (blurred real images from input_dir) ───────────
elif mode == "real":
    real_paths = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg"))
        + glob.glob(os.path.join(input_dir, "*.jpeg"))
        + glob.glob(os.path.join(input_dir, "*.png"))
        + glob.glob(os.path.join(input_dir, "*.tif"))
        + glob.glob(os.path.join(input_dir, "*.tiff"))
    )

    if not real_paths:
        raise RuntimeError(f"No real background images found in {input_dir!r}")

    selected_paths = random.sample(real_paths, k=min(num_images, len(real_paths)))

    for i, path in enumerate(tqdm(selected_paths, desc="Real background"), start=0):
        filename = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        image = cv2.resize(img, (img_size, img_size))
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        image, mask = Generate_layers(image, mask)
        # image, mask = apply_circular_mask(image, mask)
        # image = liot(image, window_size=random.choice([3, 5, 7]))
        cv2.imwrite(f"{output_dir}/images/{i:03d}_{Config_number}_{filename}", image)
        cv2.imwrite(f"{output_dir}/masks/{i:03d}_{Config_number}_{filename}", mask)

# ── Mode 3: FDA — generate mask with Generate_layers, adapt onto real image ─
elif mode == "fda":
    if not _sample_imgs:
        raise RuntimeError(f"No sample images found in {samples_dir!r}")
    for i in tqdm(range(num_images), desc="FDA real"):
        real_sample = random.choice(_sample_imgs)
        # Build crack mask on top of the real image (so layer colours match its tone)
        image = real_sample.copy()
        mask  = np.zeros((img_size, img_size), dtype=np.uint8)
        image, mask = Generate_layers(image, mask)
        # Adapt the mask back onto the (unmodified) real sample via FDA
        if np.any(mask > 0):
            l_value = round(random.uniform(*FDA_CONFIG["L_range"]), FDA_CONFIG["L_round_digits"])
            fda_img, fda_mask = generate_fda_composite(real_sample, mask, L=l_value)
        else:
            fda_img,  fda_mask = real_sample.copy(), mask
        cv2.imwrite(f"{output_dir}/images/image_{i:03d}_{Config_number}.png", fda_img)
        cv2.imwrite(f"{output_dir}/masks/mask_{i:03d}_{Config_number}.png",   mask)

else:
    raise ValueError(f"Unknown mode {mode!r}. Choose 'simple', 'real', or 'fda'.")