from PIL import Image

def remove_white_pixels(input_path, output_path, threshold=240):
    """
    Removes white (or near-white) pixels from an image by making them transparent.

    Args:
        input_path (str): Path to input PNG image
        output_path (str): Path to save output PNG image
        threshold (int): How close to white a pixel must be (0–255)
                         Higher = more aggressive removal
    """
    # Open image and convert to RGBA (adds transparency channel)
    img = Image.open(input_path).convert("RGBA")
    data = img.getdata()

    new_data = []
    for pixel in data:
        r, g, b, a = pixel

        # Check if pixel is "white enough"
        if r >= threshold and g >= threshold and b >= threshold:
            # Make transparent
            new_data.append((255, 255, 255, 0))
        else:
            # Keep original pixel
            new_data.append(pixel)

    img.putdata(new_data)
    img.save(output_path, "PNG")


# Example usage
remove_white_pixels("featureimportance.png", "output.png")