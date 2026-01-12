import numpy as np
from PIL import Image


def convert_image_to_ascii(
    image: Image.Image,
    ascii_space_width: int = 1,
) -> str:
    assert ascii_space_width is not None, "ascii_space_width must be specified"
    assert ascii_space_width > 0, "ascii_space_width must be greater than 0"
    width, height = image.size
    image = image.load()
    strings = []
    for y in range(height):
        for x in range(width):
            r, g, b, *a = image[x, y]
            a = a[0] if a else 255
            if a == 0:
                # Transparent pixel -> uncolored space
                strings.append(" " * ascii_space_width)
            else:
                # ANSI escape: background color with truecolor
                strings.append(f"\x1b[48;2;{r};{g};{b}m" + (" " * ascii_space_width) + "\x1b[0m")
        strings.append("\n")
    return "".join(strings)


def convert_image_to_halfblock(
    image: Image.Image,
    alpha_threshold: int = 64,
    use_256_colors: bool = False,
) -> str:
    """Convert image to ANSI half-block art.

    Uses ▀ (upper half block) where each character represents 2 vertical pixels.
    Foreground color = top pixel, background color = bottom pixel.

    Args:
        image: PIL Image (will be converted to RGBA)
        alpha_threshold: Alpha values >= this are visible (default 64 = 25%)
        use_256_colors: Use 256-color mode instead of truecolor (for compatibility)

    Returns:
        String with ANSI escape codes for terminal display
    """
    image = image.convert('RGBA')
    width, height = image.size
    pixels = image.load()

    # Ensure even height
    if height % 2 != 0:
        height -= 1

    lines = []
    for y in range(0, height, 2):
        line = []
        for x in range(width):
            # Top pixel
            r1, g1, b1, a1 = pixels[x, y]
            # Bottom pixel
            r2, g2, b2, a2 = pixels[x, y + 1] if y + 1 < image.height else (0, 0, 0, 0)

            # Apply alpha threshold
            top_visible = a1 >= alpha_threshold
            bot_visible = a2 >= alpha_threshold

            if use_256_colors:
                fg1 = _rgb_to_256(r1, g1, b1)
                fg2 = _rgb_to_256(r2, g2, b2)
                fg_code = lambda c: f"\x1b[38;5;{c}m"
                bg_code = lambda c: f"\x1b[48;5;{c}m"
            else:
                fg_code = lambda rgb: f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
                bg_code = lambda rgb: f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
                fg1, fg2 = (r1, g1, b1), (r2, g2, b2)

            if top_visible and bot_visible:
                # Both visible: ▀ with fg=top, bg=bottom
                line.append(f"{fg_code(fg1)}{bg_code(fg2)}▀\x1b[0m")
            elif top_visible:
                # Only top: ▀ with fg=top
                line.append(f"{fg_code(fg1)}▀\x1b[0m")
            elif bot_visible:
                # Only bottom: ▄ with fg=bottom
                line.append(f"{fg_code(fg2)}▄\x1b[0m")
            else:
                # Both transparent: space
                line.append(" ")
        lines.append("".join(line))

    return "\n".join(lines)


def _rgb_to_256(r: int, g: int, b: int) -> int:
    """Convert RGB to nearest ANSI 256 color."""
    # Check grayscale
    if abs(r - g) < 10 and abs(g - b) < 10:
        gray = (r + g + b) // 3
        if gray < 8:
            return 16
        if gray > 248:
            return 231
        return round((gray - 8) / 247 * 24) + 232

    # Color cube: 6x6x6
    return 16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)


def naive_median(X: np.ndarray) -> np.ndarray:
    """
    Returns the naive median of points in X.

    By orip, released under zlib license.
    Lightly modified for readability.
    https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

    """
    return np.median(X, axis=0)


def geometric_median(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Returns the geometric median of points in X.

    By orip, released under zlib license.
    Lightly modified for readability.
    https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points

    """
    from scipy.spatial.distance import cdist, euclidean

    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def crop_to_content(image: Image.Image) -> Image.Image:
    """
    Automatically crop an image with an alpha channel to the first and last rows and columns
    where all pixels aren't transparent.

    Args:
        image: A PIL Image object with an alpha channel (RGBA mode)

    Returns:
        A cropped PIL Image object
    """
    # Ensure the image has an alpha channel
    if image.mode != "RGBA":
        return image

    # Get alpha channel
    alpha = np.array(image.split()[3])

    # Find the bounding box of non-transparent pixels
    # Get the non-zero alpha locations
    non_transparent = np.where(alpha > 0)

    if len(non_transparent[0]) == 0:  # No non-transparent pixels found
        return image

    # Find the bounding box
    min_y, max_y = non_transparent[0].min(), non_transparent[0].max()
    min_x, max_x = non_transparent[1].min(), non_transparent[1].max()

    # Add 1 to max values because PIL's crop is inclusive of the start coordinates
    # but exclusive of the end coordinates
    # Crop the image to the bounding box
    cropped_image = image.crop((min_x, min_y, max_x + 1, max_y + 1))

    return cropped_image


def quantize_image(
    image: Image.Image,
    color_bits: int = 8,
    alpha_bits: int = 8,
    alpha_threshold: int = None,
    num_colors: int = None,
) -> Image.Image:
    """
    Quantize image colors and/or alpha channel.

    Args:
        image: PIL Image (will be converted to RGBA)
        color_bits: Bit depth per color channel (1-8). 8 = no change, 4 = 16 levels, etc.
        alpha_bits: Bit depth for alpha channel (1-8). 8 = no change.
        alpha_threshold: If set, binary alpha: >= threshold becomes 255, < threshold becomes 0
        num_colors: If set, use palette quantization to reduce to N colors (ignores color_bits)

    Returns:
        Quantized PIL Image in RGBA mode
    """
    image = image.convert('RGBA')
    arr = np.array(image, dtype=np.float32)

    # Palette-based quantization
    if num_colors is not None:
        # Use PIL's built-in quantization for palette reduction
        rgb = image.convert('RGB')
        quantized_rgb = rgb.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
        quantized_rgb = quantized_rgb.convert('RGB')
        # Restore alpha from original
        result = Image.merge('RGBA', (*quantized_rgb.split(), image.split()[3]))
        arr = np.array(result, dtype=np.float32)

    # Bit-depth quantization for colors
    if color_bits < 8 and num_colors is None:
        max_val = 2 ** color_bits - 1
        for c in range(3):  # RGB channels only
            arr[:, :, c] = np.round(arr[:, :, c] * max_val / 255) * 255 / max_val

    # Alpha processing
    if alpha_threshold is not None:
        # Binary alpha based on threshold
        arr[:, :, 3] = np.where(arr[:, :, 3] >= alpha_threshold, 255, 0)
    elif alpha_bits < 8:
        # Bit-depth quantization for alpha
        max_val = 2 ** alpha_bits - 1
        arr[:, :, 3] = np.round(arr[:, :, 3] * max_val / 255) * 255 / max_val

    return Image.fromarray(arr.astype(np.uint8), mode='RGBA')
