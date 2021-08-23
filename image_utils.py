import numpy as np
from PIL import Image

def img_format(img):
    img = np.clip(np.array(img), -1, 1)
    return ((img + 1) / 2 * 255).astype(np.uint8)

def save_image(filename, img):
    result = Image.fromarray(img_format(img))
    result.save(filename)

def save_image_grid(filename, imgs, rows, cols):
    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h, ch = imgs[0].shape
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
    
        for i, img in enumerate(imgs):
            grid.paste(Image.fromarray(img_format(img)), box=(i%cols*w, i//cols*h))
        return grid

    grid = image_grid(imgs, rows, cols)
    grid.save(filename)
