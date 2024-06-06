import os
from PIL import Image, ImageOps

def create_map_patch(path1, path2):
    xmin, ymin = 1000000000, 1000000000
    xmax, ymax = 0, 0
    for path in [path1, path2]:
        for patch_name in os.listdir(path):
            if len(patch_name[:-4]) <= 7:
                continue
            x, y = int(patch_name[:4]), int(patch_name[5:10])
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y

    patch_size=(64,64)

    map = Image.new("RGB", ((xmax-xmin) * patch_size[0], (ymax-ymin) * patch_size[1]))   

    for path in [path1, path2]:
        for patch_name in os.listdir(path):
            if len(patch_name[:-4]) <= 7:
                continue

            image_path = os.path.join(path, patch_name)
            patch_image = Image.open(image_path)

            # xtile, ytile = int(patch_name[:2])-53, 12-int(patch_name[5:7])
            xtile, ytile = int(patch_name[:4])-xmin, ymax-int(patch_name[5:10])
            
            patch_image_with_border = ImageOps.expand(patch_image, border=5, fill='red')
            patch_image = patch_image_with_border.resize(patch_size)
            map.paste(patch_image, (xtile*patch_size[0], ytile*patch_size[0]))
    print(xmin, xmax, ymin, ymax)
    map.save(f"balsfjord_map.png", format="PNG")   

if __name__ == "__main__":

    folder_path1 = "./data/original/ortophotos/train"
    folder_path2 = "./data/original/ortophotos/val2"

    create_map_patch(folder_path1, folder_path2)