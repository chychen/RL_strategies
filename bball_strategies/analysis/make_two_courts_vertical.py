from PIL import Image
import numpy as np


images = [Image.fromarray(np.asarray(Image.open("court.png"))[:,:,:3]) for _ in range(2)]


widths, heights = zip(* (i.size for i in images))

max_width = max(widths)
total_h = sum(heights)

new_img = Image.new('RGB', (max_width, total_h + 50))


y_offset = 0
for img in images:
    new_img.paste(img, (0, y_offset))
    y_offset += img.size[1] + 50

gg = np.array(new_img)

for i in range(gg.shape[0]):
    for j in range(gg.shape[1]):
        if gg[i, j, 0] == 0:
            gg[i, j, 0] = 255
            gg[i, j, 1] = 255
            gg[i, j, 2] = 255
        else:
            gg[i, j, 0] = 100
            gg[i, j, 1] = 100
            gg[i, j, 2] = 100
gg = Image.fromarray(gg)   
gg.save('vertical_both.png')