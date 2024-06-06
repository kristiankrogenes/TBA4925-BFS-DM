from PIL import Image
import numpy as np 
import os


def tif_to_png():
    tif_folder_path = './data/NorkartSOTA/tif/' 
    for tif in os.listdir(tif_folder_path):
        tif_file_path = os.path.join(tif_folder_path, tif)
        tif_image = Image.open(tif_file_path)

        tif_data = np.array(tif_image)
        print(tif_data.shape, np.min(tif_data), np.max(tif_data))
        if np.max(tif_data) > 0:
            tif_data_normalized = (tif_data - np.min(tif_data)) / (np.max(tif_data) - np.min(tif_data)) * 255
            tif_data_normalized = tif_data_normalized.astype(np.uint8)
        else:
            tif_data_normalized = tif_data
        # print(tif_data_normalized.shape, np.min(tif_data_normalized), np.max(tif_data_normalized))

        rgb_image = Image.fromarray(tif_data_normalized)

        png_folder_path = './data/NorkartSOTA/png'
        png_file_path = os.path.join(png_folder_path, tif[:-4])
        rgb_image.save(f"{png_file_path}.png")

def check_masks():
    pred_folder_path = './data/NorkartSOTA/png/' 
    labels_folder_path = './data/norkart/final/labels/' 
    label_names = os.listdir(labels_folder_path)
    for pred_name in os.listdir(pred_folder_path):
        if not pred_name[:7]+".png" in label_names:
            print(pred_name[:7])
            raise ValueError(f"Prediction {pred_name} not in labels.")


def filter_black_images():
    pred_folder_path = './data/NorkartSOTA/png/' 
    labels_folder_path = './data/norkart/final/labels/' 
    label_names = os.listdir(labels_folder_path)
    for label_name in os.listdir(labels_folder_path):
        label_array = np.asarray(Image.open("./data/norkart/final/labels/"+label_name))
        if np.max(label_array) > 0:
            # label_image = Image.fromarray(label_array)
            # label_image.save(f"./data/norkart/final/labels_filtered/{label_name}")
            # pred_image = Image.open("./data/NorkartSOTA/png/"+label_name[:-4]+"_prediction.png")
            # pred_image.save(f"./data/NorkartSOTA/final/{label_name}")
            ortho_image = Image.open("./data/norkart/final/ortho/"+label_name)
            ortho_image.save(f"./data/original/ortophotos/val/{label_name}")
# filter_black_images()


print("ORTHO", len(os.listdir("./data/norkart/final/ortho/")))
print("ORTHO", len(os.listdir('./data/original/ortophotos/val')))
print("NORKART SEGEMENTATIONS", len(os.listdir('./data/processed/predictions/val')))
print("LABELS", len(os.listdir('./data/original/labels/val')))

# 25832_563000.0_6623000.0_100.0_100.0