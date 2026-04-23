import numpy as np
import cv2
import random
import os
from tqdm import tqdm
from skimage import exposure

# Parameters
img_size = 800
output_dir = "Data/Generalized_dataset"
input_dir = "./output_images/"  # Directory with .tif images
background = False
num_images = 300
thickness_range = (1, 9)
color_range = (0, 149)

# Create output directories
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/masks", exist_ok=True)

# ===== BACKGROUND GENERATION FUNCTIONS =====

def generate_perlin_texture(size, octaves=4, scale=100):
    """Generate Perlin-like noise texture using multiple frequency layers"""
    texture = np.zeros((size, size), dtype=np.float32)
    
    for octave in range(octaves):
        freq = 2 ** octave
        amplitude = 1.0 / (2 ** octave)
        
        # Generate noise for this octave
        noise_size = size // freq + 1
        noise = np.random.rand(noise_size, noise_size)
        
        # Upsample to full size
        noise_upsampled = cv2.resize(noise, (size, size), interpolation=cv2.INTER_LINEAR)
        texture += noise_upsampled * amplitude
    
    # Normalize to 0-255
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    return (texture * 255).astype(np.uint8)


def generate_concrete_texture(size):
    """Generate concrete-like texture with varied grain"""
    # Base layer
    base = np.random.randint(100, 200, (size, size), dtype=np.uint8)
    
    # Add fine grain
    fine_grain = np.random.normal(0, 15, (size, size))
    base = np.clip(base + fine_grain, 0, 255).astype(np.uint8)
    
    # Add medium grain (small aggregates)
    for _ in range(random.randint(200, 500)):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        radius = random.randint(1, 4)
        intensity = random.randint(-30, 30)
        new_color = int(np.clip(int(base[y, x]) + intensity, 0, 255))
        cv2.circle(base, (x, y), radius, new_color, -1)
    
    # Smooth slightly for realism
    base = cv2.GaussianBlur(base, (3, 3), 0.5)
    
    return base


def generate_asphalt_texture(size):
    """Generate asphalt-like texture"""
    # Dark base
    base = np.random.randint(40, 90, (size, size), dtype=np.uint8)
    
    # Add aggregate stones
    for _ in range(random.randint(100, 300)):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        radius = random.randint(1, 3)
        intensity = random.randint(80, 140)
        cv2.circle(base, (x, y), radius, intensity, -1)
    
    # Add fine texture
    noise = np.random.normal(0, 10, (size, size))
    base = np.clip(base + noise, 0, 255).astype(np.uint8)
    
    # Blur for smooth asphalt look
    base = cv2.GaussianBlur(base, (5, 5), 1)
    
    return base


def generate_marble_texture(size):
    """Generate marble-like texture with veins"""
    # Base color variation
    base = generate_perlin_texture(size, octaves=3, scale=150)
    base = np.clip(base * 0.8 + 120, 100, 220).astype(np.uint8)
    
    # Add veins
    for _ in range(random.randint(3, 8)):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        angle = random.uniform(0, 2 * np.pi)
        
        for i in range(random.randint(50, 200)):
            angle += random.uniform(-0.3, 0.3)
            step = random.uniform(2, 5)
            new_x = int(np.clip(x + step * np.cos(angle), 0, size-1))
            new_y = int(np.clip(y + step * np.sin(angle), 0, size-1))
            
            thickness = random.randint(1, 3)
            intensity = random.randint(80, 120)
            cv2.line(base, (x, y), (new_x, new_y), intensity, thickness)
            
            x, y = new_x, new_y
            
            if x <= 0 or x >= size-1 or y <= 0 or y >= size-1:
                break
    
    return base


def generate_textured_background(size):
    """Generate diverse textured background"""
    texture_type = random.choice(['concrete', 'asphalt', 'marble', 'perlin', 'mixed'])
    
    if texture_type == 'concrete':
        background = generate_concrete_texture(size)
    elif texture_type == 'asphalt':
        background = generate_asphalt_texture(size)
    elif texture_type == 'marble':
        background = generate_marble_texture(size)
    elif texture_type == 'perlin':
        background = generate_perlin_texture(size, octaves=random.randint(3, 6))
        # Adjust brightness randomly
        background = np.clip(background * random.uniform(0.6, 1.2) + random.randint(-30, 30), 0, 255).astype(np.uint8)
    else:  # mixed
        # Blend two textures
        tex1 = generate_concrete_texture(size)
        tex2 = generate_perlin_texture(size)
        alpha = random.uniform(0.3, 0.7)
        background = cv2.addWeighted(tex1, alpha, tex2, 1-alpha, 0)
    
    # Add subtle lighting variation
    if random.random() < 0.5:
        # Create gradient overlay
        gradient = np.linspace(0, 1, size)
        if random.random() < 0.5:
            gradient = gradient.reshape(1, -1).repeat(size, axis=0)  # Horizontal
        else:
            gradient = gradient.reshape(-1, 1).repeat(size, axis=1)  # Vertical
        
        gradient_effect = (gradient * random.randint(20, 50) - random.randint(10, 25)).astype(np.int16)
        background = np.clip(background.astype(np.int16) + gradient_effect, 0, 255).astype(np.uint8)
    
    return background

# Range for step size in random walk


# Generate realistic fracture using improved random walk with branching
def realistic_crack_fracture(size, gray_color):
    """
    Generate realistic crack using improved random walk with branching.
    Creates natural-looking cracks with variable width and branching patterns.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Start from random position
    x, y = random.randint(size // 4, 3 * size // 4), random.randint(size // 4, 3 * size // 4)
    
    # Main crack parameters
    main_length = random.randint(150, 600)
    base_thickness = random.randint(*thickness_range)
    
    # Direction bias for more natural cracks
    preferred_angle = random.uniform(0, 2 * np.pi)
    angle_deviation = 0.3  # How much the crack can deviate from preferred direction
    
    points = [(x, y)]
    
    # Generate main crack path
    for i in range(main_length):
        # Add some randomness to angle but bias towards preferred direction
        angle = preferred_angle + random.uniform(-angle_deviation, angle_deviation)
        preferred_angle += random.uniform(-0.1, 0.1)  # Slowly change preferred direction
        
        # Variable step size for more natural appearance
        step_size = random.uniform(1, 3)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_x = int(np.clip(x + dx, 0, size - 1))
        new_y = int(np.clip(y + dy, 0, size - 1))
        
        # Variable thickness along crack
        thickness_variation = random.uniform(0.5, 1.5)
        current_thickness = max(1, int(base_thickness * thickness_variation))
        
        cv2.line(img, (x, y), (new_x, new_y), gray_color, current_thickness)
        cv2.line(mask, (x, y), (new_x, new_y), 255, current_thickness)
        
        # Add branching with low probability
        if random.random() < 0.01 and i > 20:  # 1% chance after initial segment
            branch_angle = angle + random.choice([-1, 1]) * random.uniform(0.3, 0.8)
            branch_length = random.randint(20, 100)
            branch_thickness = max(1, current_thickness - random.randint(1, 2))
            
            bx, by = new_x, new_y
            for _ in range(branch_length):
                branch_angle += random.uniform(-0.2, 0.2)
                bstep = random.uniform(0.8, 2)
                bdx = bstep * np.cos(branch_angle)
                bdy = bstep * np.sin(branch_angle)
                
                bnew_x = int(np.clip(bx + bdx, 0, size - 1))
                bnew_y = int(np.clip(by + bdy, 0, size - 1))
                
                cv2.line(img, (bx, by), (bnew_x, bnew_y), gray_color, branch_thickness)
                cv2.line(mask, (bx, by), (bnew_x, bnew_y), 255, branch_thickness)
                
                bx, by = bnew_x, bnew_y
                
                # Stop if we hit edge
                if bx <= 1 or bx >= size - 2 or by <= 1 or by >= size - 2:
                    break
        
        x, y = new_x, new_y
        points.append((x, y))
        
        # Stop if we hit edge
        if x <= 1 or x >= size - 2 or y <= 1 or y >= size - 2:
            break
    
    return img, mask


# Generate fracture using random walk
def random_walk_fracture(size, gray_color):
    step_range = (-3, 3)
    img = np.zeros((size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    x, y = random.randint(0, size - 1), random.randint(0, size - 1)
    length = random.randint(100, 500)
    thickness = random.randint(*thickness_range)
    for _ in range(length):
        dx, dy = random.uniform(*step_range), random.uniform(*step_range)
        new_x = int(np.clip(x + dx, 0, size - 1))
        new_y = int(np.clip(y + dy, 0, size - 1))
        cv2.line(img, (x, y), (new_x, new_y), gray_color, thickness)
        cv2.line(mask, (x, y), (new_x, new_y), 255, thickness)
        x, y = new_x, new_y
    return img, mask

# Generate fracture using a straight line
def random_line_fracture(size, gray_color):
    img = np.zeros((size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    thickness = random.randint(*thickness_range)
    x1, y1 = random.randint(0, size - 1), random.randint(0, size - 1)
    angle = random.uniform(0, 360)
    length = random.randint(size // 10, size -1)
    x2 = int(np.clip(x1 + length * np.cos(np.radians(angle)), 0, size - 1))
    y2 = int(np.clip(y1 + length * np.sin(np.radians(angle)), 0, size - 1))
    cv2.line(img, (x1, y1), (x2, y2), gray_color, thickness)
    # cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    return img, mask

def draw_branch(img, mask, x, y, angle, length, depth, thickness, color):
    if depth <= 0:
        return img, mask
    length_range = (10, 50)
    angle_variation = 0.5  # Radians
    end_x = int(x + length * np.cos(angle))
    end_y = int(y + length * np.sin(angle))
    cv2.line(img, (x, y), (end_x, end_y), color, thickness)
    cv2.line(mask, (x, y), (end_x, end_y), 255, thickness)
    
    num_children = random.choice([1, 2])  # Can branch into 1 or 2
    for _ in range(num_children):
        new_angle = angle + random.uniform(-angle_variation, angle_variation)
        new_length = random.randint(*length_range)
        draw_branch(img, mask, end_x, end_y, new_angle, new_length, depth - 1, thickness, color)

def draw_shape(image, mask, shape_type, gray):
    img_size = image.shape[0]  # Ensure img_size is defined based on input image
    shape_img = np.zeros_like(image, dtype=np.uint8)
    shape_mask = np.zeros_like(mask, dtype=np.uint8)

    if shape_type == 'smallcircle':
        center = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
        radius = random.randint(1, 7)
        cv2.circle(shape_img, center, radius, gray, -1)
        cv2.circle(shape_mask, center, radius, 255, -1)

    elif shape_type == 'spot':
        # Irregular organic spot: distorted ellipse polygon
        cx = random.randint(0, img_size - 1)
        cy = random.randint(0, img_size - 1)
        rx = random.randint(2, 18)
        ry = int(rx * random.uniform(0.4, 1.6))
        rot = random.uniform(0, 2 * np.pi)
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        num_pts = random.randint(10, 22)
        pts = []
        for k in range(num_pts):
            theta = 2 * np.pi * k / num_pts
            # Radial perturbation for irregular edge
            r = 1.0 + random.uniform(-0.4, 0.4)
            ex = r * rx * np.cos(theta)
            ey = r * ry * np.sin(theta)
            px = int(np.clip(cx + ex * cos_r - ey * sin_r, 0, img_size - 1))
            py = int(np.clip(cy + ex * sin_r + ey * cos_r, 0, img_size - 1))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(shape_img, [pts], gray)
        cv2.fillPoly(shape_mask, [pts], 255)


    if shape_type == 'circle':
        center = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
        radius = random.randint(10, img_size // 6)
        cv2.circle(shape_img, center, radius, gray, -1)
        cv2.circle(shape_mask, center, radius, 255, -1)

    elif shape_type == 'rectangle':
        pt1 = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
        pt2 = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
        cv2.rectangle(shape_img, pt1, pt2, gray, -1)
        cv2.rectangle(shape_mask, pt1, pt2, 255, -1)

    elif shape_type == 'square':
        side = random.randint(1, img_size // 4)
        x = random.randint(0, img_size - side)
        y = random.randint(0, img_size - side)
        pt1 = (x, y)
        pt2 = (x + side, y + side)
        cv2.rectangle(shape_img, pt1, pt2, gray, -1)
        cv2.rectangle(shape_mask, pt1, pt2, 255, -1)

    elif shape_type == 'triangle':
        pts = np.array([
            [random.randint(0, img_size - 1), random.randint(0, img_size - 1)],
            [random.randint(0, img_size - 1), random.randint(0, img_size - 1)],
            [random.randint(0, img_size - 1), random.randint(0, img_size - 1)]
        ], np.int32)
        cv2.fillPoly(shape_img, [pts], gray)
        cv2.fillPoly(shape_mask, [pts], 255)

    # Combine shape with original image
    image = np.where(shape_img > 0, shape_img, image)
    # mask = cv2.bitwise_or(mask, shape_mask)  # Uncomment if needed

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

def Generate_layers(image, mask):
    # Random number of fractures and shapes
    num_fractures = random.randint(0, 10)
    num_shapes = random.randint(0, 10)
    num_smallcircles = random.randint(0, 1000)

    # Create drawing plan
    draw_steps = ['shape'] * num_smallcircles
    # random.shuffle(draw_steps)

    for step in draw_steps:
        if step == 'shape':
            shape = random.choice(['smallcircle', 'spot'])
            gray = random.randint(0, 255)
            image, mask = draw_shape(image, mask, shape, gray)

    # Create drawing plan
    draw_steps = ['fracture'] * num_fractures
    # random.shuffle(draw_steps)

    for step in draw_steps:
        if step == 'fracture':
            gray_color = random.randint(*color_range)
            
            # Use realistic crack generation method
            crack_type = random.random()
            if crack_type < 0.9:  # 60% realistic branching cracks
                obj, obj_mask = realistic_crack_fracture(img_size, gray_color)
            elif crack_type < 0:  # 25% random walk
                obj, obj_mask = random_walk_fracture(img_size, gray_color)
            else:  # 15% straight lines
                obj, obj_mask = random_line_fracture(img_size, gray_color)

            image = np.where(obj > 0, obj, image)
            mask = cv2.bitwise_or(mask, obj_mask)

        elif step == 'shape':
            shape = random.choice(['circle', 'rectangle', 'square', 'triangle'])
            gray = random.randint(0, 255)
            image, mask = draw_shape(image, mask, shape, gray)

    for i in range(random.randint(0,0)):
        root_x, root_y = random.randint(100, 150), random.randint(200, 300)
        initial_angle = random.uniform(-np.pi/4, np.pi/4)
        thickness = random.randint(*thickness_range)
        gray_color = random.randint(*color_range)
        draw_branch(image, mask, root_x, root_y, initial_angle, 10, random.randint(3, 10), thickness, gray_color)


    # Optionally apply Gaussian blur
    if random.random() < 0:  # 70% chance
        ksize = random.choice([3, 5, 7, 11, 13])
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=random.randint(1,4))

    if random.random() < 0.5:
        image = add_gaussian_noise(image)
    if random.random() < 0.5:
        image = add_poisson_noise(image)
    if random.random() < 0.5:
        image = add_salt_pepper_noise(image)
    if random.random() < 0.5:
        image = add_speckle_noise(image)

    return image, mask

# --- Main Processing Loop ---
if background:
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(".jpg"):
            for i in range(1):
                path = os.path.join(input_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Resize to 448x448
                image = cv2.resize(img, (img_size, img_size))
                mask = np.zeros((img_size, img_size), dtype=np.uint8)
                image, mask = Generate_layers(image, mask)
                # Apply circular mask
                image, mask = apply_circular_mask(image, mask)
                # image = liot(image, window_size=random.choice([3,5,7])) #, random.choice([3,5,7])
                # Save result
                cv2.imwrite(f"{output_dir}/images/{i}_{filename}", image)
                cv2.imwrite(f"{output_dir}/masks/{i}_{filename}", mask)
else:
    for i in tqdm(range(num_images), desc="Generating realistic crack dataset with diverse backgrounds"):
        # Generate diverse synthetic background instead of solid color
        image = generate_textured_background(img_size)
        
        # Adjust color range based on background brightness
        bg_mean = np.mean(image)
        color_range = (max(0, int(bg_mean - 80)), int(bg_mean - 10))
        
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        image, mask = Generate_layers(image, mask)
        # Apply circular mask
        image, mask = apply_circular_mask(image, mask)
        # image = liot(image)
        # Save final image and mask
        cv2.imwrite(f"{output_dir}/images/image_{i:03d}.png", image)
        cv2.imwrite(f"{output_dir}/masks/mask_{i:03d}.png", mask)