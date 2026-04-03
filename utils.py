"""
utils.py — shared helpers for the Hand Gesture Recognition project.

PIPELINE CONTRACT (must match train.py exactly):
  • Load image as RGB (3 channels, no grayscale conversion).
  • Resize to (128, 128).
  • Keep pixel values in [0, 255] as uint8 — the model's first layer
    calls tf.keras.applications.mobilenet_v2.preprocess_input, which
    maps [0, 255] → [-1, 1] internally.  Do NOT divide by 255 here.
"""

import cv2
import numpy as np
from PIL import Image


# ─── Gesture metadata ─────────────────────────────────────────────────────────

GESTURES = [
    "01_palm", "02_I", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down",
]

DISPLAY_NAMES = {
    "01_palm":       "Palm",
    "02_I":          "I Shape",
    "03_fist":       "Fist",
    "04_fist_moved": "Fist Moved",
    "05_thumb":      "Thumb",
    "06_index":      "Index",
    "07_ok":         "OK",
    "08_palm_moved": "Palm Moved",
    "09_c":          "C Shape",
    "10_down":       "Down",
}

GESTURE_EMOJI = {
    "01_palm":       "✋",
    "02_I":          "🇮",
    "03_fist":       "✊",
    "04_fist_moved": "🏃",
    "05_thumb":      "👍",
    "06_index":      "☝️",
    "07_ok":         "👌",
    "08_palm_moved": "🌀",
    "09_c":          "©️",
    "10_down":       "👇",
}


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(
    image,
    target_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Preprocess an image for model inference.

    Accepts
    -------
    image : str | PIL.Image.Image | np.ndarray
        • str          — file path
        • PIL.Image    — from st.file_uploader / Image.open
        • np.ndarray   — OpenCV frame (BGR, BGRA, or grayscale)

    Returns
    -------
    np.ndarray, shape (1, H, W, 3), dtype uint8
        Ready to pass directly to model.predict().
        The model's preprocess_input layer handles [0-255] → [-1, 1].
    """

    # ── 1. Decode to a numpy RGB array ────────────────────────────────────
    if isinstance(image, str):
        bgr = cv2.imread(image)
        if bgr is None:
            raise ValueError(f"Cannot load image from path: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif isinstance(image, Image.Image):
        # PIL: convert to RGB regardless of original mode (L, RGBA, P, …)
        rgb = np.array(image.convert('RGB'))

    elif isinstance(image, np.ndarray):
        if image.ndim == 2:                        # grayscale → RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:                  # BGRA → RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:                                      # BGR → RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ── 2. Resize ──────────────────────────────────────────────────────────
    rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LANCZOS4)

    # ── 3. Add batch dimension ─────────────────────────────────────────────
    #    Shape: (1, H, W, 3), dtype uint8 — matches training distribution.
    return np.expand_dims(rgb.astype(np.uint8), axis=0)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_gesture_emoji(gesture_key: str) -> str:
    """Return an emoji for the given gesture key (e.g. '01_palm')."""
    return GESTURE_EMOJI.get(gesture_key, "🖐️")


def get_display_name(gesture_key: str) -> str:
    """Return the human-readable name for a gesture key."""
    return DISPLAY_NAMES.get(gesture_key, gesture_key)


def decode_prediction(
    probabilities: np.ndarray,
    label_encoder,
    top_k: int = 3,
) -> list[dict]:
    """
    Convert raw softmax output to a ranked list of predictions.

    Parameters
    ----------
    probabilities : np.ndarray, shape (num_classes,) or (1, num_classes)
    label_encoder : sklearn.preprocessing.LabelEncoder
    top_k         : number of top results to return

    Returns
    -------
    list of dicts, each with keys:
        gesture_key   : str   e.g. "01_palm"
        display_name  : str   e.g. "Palm"
        emoji         : str   e.g. "✋"
        confidence    : float e.g. 0.9842
        confidence_pct: str   e.g. "98.42%"
    """
    probs = np.squeeze(probabilities)          # (num_classes,)
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in top_indices:
        key = label_encoder.inverse_transform([idx])[0]
        results.append({
            "gesture_key":    key,
            "display_name":   get_display_name(key),
            "emoji":          get_gesture_emoji(key),
            "confidence":     float(probs[idx]),
            "confidence_pct": f"{probs[idx]*100:.2f}%",
        })
    return results