from PIL import Image

def make_seamless_image(image1, image2, direction, blend_size):
    """
    Stitches two images together either horizontally or vertically
    with a linear alpha blend transition.
    """
    # Ensure images are in RGB mode
    image1, image2 = image1.convert("RGB"), image2.convert("RGB")
    blend_size = int(blend_size)
    
    if direction == "Side by Side":
        # 1. Normalize Heights
        h = min(image1.height, image2.height)
        image1 = image1.resize((int(image1.width * h / image1.height), h))
        image2 = image2.resize((int(image2.width * h / image2.height), h))
        
        # 2. Create Canvas (Width is sum of both minus the overlap)
        canvas = Image.new("RGB", (image1.width + image2.width - blend_size, h))
        canvas.paste(image1, (0, 0))
        
        # 3. Perform Linear Blending in the overlap zone
        for x in range(blend_size):
            alpha = x / blend_size
            # Cut a 1-pixel slice from the end of image1
            col1 = image1.crop((image1.width - blend_size + x, 0, image1.width - blend_size + x + 1, h))
            # Cut a 1-pixel slice from the start of image2
            col2 = image2.crop((x, 0, x + 1, h))
            # Blend and paste into the canvas
            canvas.paste(Image.blend(col1, col2, alpha), (image1.width - blend_size + x, 0))
        
        # 4. Paste the remainder of image2
        canvas.paste(image2.crop((blend_size, 0, image2.width, h)), (image1.width, 0))
        return canvas, "✅ Side-by-side stitch created."
    
    else: # Top and Bottom
        # 1. Normalize Widths
        w = min(image1.width, image2.width)
        image1 = image1.resize((w, int(image1.height * w / image1.width)))
        image2 = image2.resize((w, int(image2.height * w / image2.width)))
        
        # 2. Create Canvas
        canvas = Image.new("RGB", (w, image1.height + image2.height - blend_size))
        canvas.paste(image1, (0, 0))
        
        # 3. Perform Linear Blending in the overlap zone
        for y in range(blend_size):
            alpha = y / blend_size
            # Cut a 1-pixel slice from the bottom of image1
            row1 = image1.crop((0, image1.height - blend_size + y, w, image1.height - blend_size + y + 1))
            # Cut a 1-pixel slice from the top of image2
            row2 = image2.crop((0, y, w, y + 1))
            # Blend and paste
            canvas.paste(Image.blend(row1, row2, alpha), (0, image1.height - blend_size + y))
        
        # 4. Paste the remainder of image2
        canvas.paste(image2.crop((0, blend_size, w, image2.height)), (0, image1.height))
        return canvas, "✅ Top-bottom stitch created."
