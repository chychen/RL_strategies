from PIL import Image

images = [Image.open("court.png") for _ in range(2)]

widths, heights = zip(* (i.size for i in images))

total_width = sum(widths)
max_h = max(heights)

new_img = Image.new('RGB', (total_width + 100, max_h))

print(total_width, max_h)

x_offset = 0
for img in images:
    new_img.paste(img, (x_offset, 0))
    x_offset += img.size[0] + 100

new_img.save('court_both.png')