
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

IMAGE_SIZE = (512, 512)
IMAGE_DIR = Path("Data/Source/Images")
MASK_DIR = Path("Data/Source/Masks")
OUTPUT_DIR = Path("Data/Preprocessed/Images")
MASK_OUTPUT_DIR = Path("Data/Preprocessed/Masks")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def get_retinal_mask(image):
    green = image[:, :, 1]
    blurred = cv2.GaussianBlur(green, (15, 15), 0)
    _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

def extract_retinal_field(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def resize_image(image, size=IMAGE_SIZE, interpolation=cv2.INTER_AREA):
    return cv2.resize(image, size, interpolation=interpolation)

def correct_illumination(image, mask, epsilon=1e-6):
    image = image.astype(np.float32)
    illumination = cv2.GaussianBlur(image, (0, 0), sigmaX=25, sigmaY=25)
    illumination = np.maximum(illumination, epsilon)
    corrected = image / illumination
    for channel in range(3):
        values = corrected[:, :, channel][mask > 0]
        if values.size:
            corrected[:, :, channel] /= max(float(np.mean(values)), epsilon)
    corrected = np.clip(corrected * 128.0, 0, 255).astype(np.uint8)
    return cv2.bitwise_and(corrected, corrected, mask=mask)

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(lightness)
    result = cv2.merge((enhanced, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def preprocess_image(image):
    if image is None:
        raise ValueError("Unable to load fundus image")
    mask = get_retinal_mask(image)
    retinal = extract_retinal_field(image, mask)
    retinal = resize_image(retinal)
    mask = resize_image(mask, interpolation=cv2.INTER_NEAREST)
    retinal = cv2.bitwise_and(retinal, retinal, mask=mask)
    retinal = correct_illumination(retinal, mask)
    retinal = enhance_contrast(retinal)
    retinal = cv2.bitwise_and(retinal, retinal, mask=mask)
    return retinal, mask

def preprocess_mask(mask):
    if mask is None:
        raise ValueError("Unable to load segmentation mask")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return resize_image(mask, interpolation=cv2.INTER_NEAREST)

def save_image(image, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to save image: {output_path}")

def process_dataset(image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR, mask_dir=None, mask_output_dir=MASK_OUTPUT_DIR):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    mask_dir = Path(mask_dir) if mask_dir is not None else None
    mask_output_dir = Path(mask_output_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")
    successful, failed = 0, []
    for image_path in tqdm(image_paths, desc="Preprocessing fundus images"):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            processed, _ = preprocess_image(image)
            relative_path = image_path.relative_to(image_dir).with_suffix(".png")
            save_image(processed, output_dir / relative_path)
            if mask_dir is not None:
                mask_path = next((mask_dir / image_path.relative_to(image_dir).with_name(image_path.stem + ext) for ext in VALID_EXTENSIONS if (mask_dir / image_path.relative_to(image_dir).with_name(image_path.stem + ext)).exists()), None)
                if mask_path is not None:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    processed_mask = preprocess_mask(mask)
                    mask_relative = mask_path.relative_to(mask_dir).with_suffix(".png")
                    save_image(processed_mask, mask_output_dir / mask_relative)
            successful += 1
        except Exception as error:
            failed.append((str(image_path), str(error)))
    print(f"Successfully processed: {successful}/{len(image_paths)}")
    if failed:
        print(f"Failed images: {len(failed)}")
        for path, error in failed:
            print(f"{path}: {error}")
    return successful, failed

if __name__ == "__main__":
    process_dataset(mask_dir=MASK_DIR if MASK_DIR.exists() else None)
