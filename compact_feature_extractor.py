import cv2
import numpy as np
from scipy.fft import dct
import os
import json
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


class CompactFeatureExtractor:
    def __init__(self, scd_bins: int = 8, cld_grid: Tuple[int, int] = (8, 8)):
        self.scd_bins = scd_bins
        self.cld_grid = cld_grid

    def _quantize_color(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = np.floor(hsv[:, :, 0] * self.scd_bins / 180).astype(np.uint8)
        s = np.floor(hsv[:, :, 1] * self.scd_bins / 256).astype(np.uint8)
        v = np.floor(hsv[:, :, 2] * self.scd_bins / 256).astype(np.uint8)
        h = np.clip(h, 0, self.scd_bins - 1)
        s = np.clip(s, 0, self.scd_bins - 1)
        v = np.clip(v, 0, self.scd_bins - 1)
        return np.stack([h, s, v], axis=2)

    def extract_scd(self, img: np.ndarray) -> np.ndarray:
        quantized = self._quantize_color(img)
        hist = np.zeros((self.scd_bins, self.scd_bins, self.scd_bins), dtype=np.float32)
        for i in range(quantized.shape[0]):
            for j in range(quantized.shape[1]):
                h, s, v = quantized[i, j]
                hist[h, s, v] += 1
        hist = hist.flatten()
        hist /= (np.sum(hist) + 1e-6)
        return hist

    def extract_cld(self, img: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img, (self.cld_grid[1], self.cld_grid[0]))
        ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        def zigzag_scan(matrix):
            rows, cols = matrix.shape
            result = []
            for i in range(rows + cols - 1):
                if i % 2 == 0:
                    for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                        result.append(matrix[j, i - j])
                else:
                    for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                        result.append(matrix[j, i - j])
            return np.array(result)

        y_dct = dct(dct(y.T).T)
        cr_dct = dct(dct(cr.T).T)
        cb_dct = dct(dct(cb.T).T)

        y_coeffs = zigzag_scan(y_dct)[:64]
        cr_coeffs = zigzag_scan(cr_dct)[:64]
        cb_coeffs = zigzag_scan(cb_dct)[:64]

        return np.concatenate([y_coeffs, cr_coeffs, cb_coeffs])

    def extract_features_dict(self, img_path: str) -> Union[Dict[str, List[float]], None]:
        # for JSON saving
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            scd = self.extract_scd(img)
            cld = self.extract_cld(img)
            return {'scd': scd.tolist(), 'cld': cld.tolist()}
        except Exception as e:
            print(f"Error: {img_path} - {str(e)}")
            return None

    def extract_features(self, img_path: str) -> Union[np.ndarray, None]:
        # for RF classifier
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            scd = self.extract_scd(img)
            cld = self.extract_cld(img)
            return np.concatenate([scd, cld])
        except Exception as e:
            print(f"Error: {img_path} - {str(e)}")
            return None




def process_image(img_path: str) -> Tuple[str, Union[Dict[str, List[float]], None]]:
    extractor = CompactFeatureExtractor()
    features = extractor.extract_features(img_path)
    return img_path, features


def load_existing_features(path: str) -> Dict[str, Dict[str, List[float]]]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, features: Dict[str, Dict[str, List[float]]]):
    with open(path, 'w') as f:
        json.dump(features, f)


if __name__ == "__main__":
    import time

    output_file = "mini_phishing_features.json"
    existing = load_existing_features(output_file)
    print(f"Loaded {len(existing)} existing features.")

    all_images = [
        os.path.join(root, file)
        for root, _, files in os.walk("mini_phishing")
        for file in files
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    to_process = [img for img in all_images if img not in existing]
    print(f"Found {len(to_process)} new images to process.")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=3) as executor:  # use 2 to stay safe on your CPU
        futures = {executor.submit(process_image, img): img for img in to_process}
        for future in tqdm(as_completed(futures), total=len(futures)):
            img_path, features = future.result()
            if features is not None:
                existing[img_path] = features

    save_checkpoint(output_file, existing)
    elapsed_time = time.time() - start_time
    print(f"Finished in {elapsed_time:.2f} seconds.")
