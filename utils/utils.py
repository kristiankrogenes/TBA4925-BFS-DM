import os
import argparse

from torchvision import transforms
import numpy as np
from PIL import Image
import csv
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
def calculate_metrics(preds, labels, verbose=False):

    average_accuracy = []
    average_precision = []
    average_recall = []
    average_f1 = []
    average_iou = []

    for index, (pred, label) in enumerate(zip(preds, labels)):

        pred[pred > 0] = 1
        pred[pred <= 0] = 0
        label[label > 0] = 1
        label[label <= 0] = 0

        pred_flat = pred.flatten()
        label_flat = label.flatten()

        accuracy = accuracy_score(label_flat, pred_flat)
        precision = precision_score(label_flat, pred_flat, zero_division=0)
        recall = recall_score(label_flat, pred_flat, zero_division=0)
        f1 = f1_score(label_flat, pred_flat, zero_division=0)
        iou = jaccard_score(label_flat, pred_flat, zero_division=0)

        average_accuracy.append(accuracy)
        average_precision.append(precision)
        average_recall.append(recall)
        average_f1.append(f1)
        average_iou.append(iou)

    if verbose:
        print(f"Metrics for {preds.__len__()} predictions ===================")
        print("Accuracy:", sum(average_accuracy) / max(1, len(average_accuracy)))
        print("Precision:", sum(average_precision) / max(1, len(average_precision)))
        print("Recall:", sum(average_recall) / max(1, len(average_recall)))
        print("F1:", sum(average_f1) / max(1, len(average_f1)))
        print("IoU:", sum(average_iou) / len(average_iou))

def transform_model_output_to_image(out_tensor):

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    return reverse_transforms(out_tensor)

def compare_inference_pred_label_orto(model, image_code):
    if not model in os.listdir("./outputs/inference"):
        raise ValueError("No existing inference on this model.")
    elif not f"{image_code}.png" in os.listdir(f"./outputs/inference/{model}"):
        raise ValueError("The model has not predicted this image code.") 
    
    image1 = Image.open(f"./outputs/inference/{model}/{image_code}.png")
    image2 = Image.open(f"./data/processed/predictions/val/{image_code}.png")
    image3 = Image.open(f"./data/original/ortophotos/val/{image_code}.png")
    image4 = Image.open(f"./data/original/labels/val/{image_code}.png")

    width, height = image1.size

    image2 = image2.resize((width, height))
    image3 = image3.resize((width, height))
    image4 = image4.resize((width, height))

    image1_array = np.asarray(image1)
    image1_array_writable = image1_array.copy()
    image1_tensor = torch.FloatTensor(image1_array_writable)
    image1_tensor = image1_tensor / 255

    image4_array = np.asarray(image4)
    image4_array_writable = image4_array.copy()
    image4_tensor = torch.FloatTensor(image4_array_writable)
    image4_tensor = image4_tensor / 255

    calculate_metrics(image1_tensor, image4_tensor, verbose=True)

    combined_image = Image.new("RGB", (3 * width, 1 * height))

    combined_image.paste(image3, (0, 0))
    combined_image.paste(image2, (width, 0))
    # combined_image.paste(image3, (0, height))
    combined_image.paste(image4, (width*2, 0))

    folder_path = os.path.join("./outputs/comparisons", model)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    combined_image_path = os.path.join(folder_path, image_code)
    combined_image.save(f"{combined_image_path}.png", format="PNG")

def add_row_to_csv(file_path, new_row):

    existing_data = []
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        existing_data.append(header)
        for row in reader:
            existing_data.append(row)

    existing_data.append(new_row)

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(existing_data)


def check_parameters_for_training(epochs, size, batch_size, timesteps, parameterization, condition_type, schedule, model_path, model_name):
    if epochs == None or epochs < 1:
        raise ValueError("Number of epochs must be given and be a positive number.")
    elif size < 1:
        raise ValueError("Size must be positive.")
    elif batch_size < 1:
        raise ValueError("Batch size must be positive.")
    elif timesteps < 1:
        raise ValueError("Number of timesteps must be positive.")
    elif not parameterization in ["eps", "x0", "v"]:
        raise ValueError("Type of parameterization is not valid.")
    elif not condition_type in ["v1", "v2", "v3", None]:
        raise ValueError("Condition type is not valid.")  
    elif not schedule in ["linear", "cosine", "softplus"]:
        raise ValueError("Type of schedule is not valid.")  
    elif not model_path==None and not os.path.exists(model_path):
        raise ValueError(f"Model {model_path} requested does not exist.")
    # elif model_name in os.listdir("./checkpoints") and model_path is None:
    #     raise ValueError(f"Model {model_name} already exists, choose another name.")
    return True

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Parameters to compare images.")
    # parser.add_argument('--model', type=str, help="Name of a trained model.")
    # parser.add_argument('--image_code', type=str, help="Image code of a given image.")
    # args = parser.parse_args()

    # compare_inference_pred_label_orto(args.model, args.image_code)

    # image1 = Image.open(f"./outputs/forward/softplus/3392_10912/noisesample_t0.png")

    # inf = 1
    sch = "cond3"
    images = [None for _ in range(11)]
    folder_path = f"./subrev/experiment3/{sch}/55_4112/"
    for im_name in os.listdir(folder_path):
        image = Image.open(f"{folder_path}{im_name}")
        # image = image.resize((128,128))
        print(im_name)
        num = int(im_name[:-4].split("_")[1])
        print(num)
        images[int(num/100)] = image
        # images.append(image)
    width, height = images[0].size
    images.reverse()

    combined_image = Image.new("RGB", (len(images) * width, 1 * height))

    for i in range(len(images)):
        combined_image.paste(images[i], (i*width, 0))

    combined_image.save(f"./subrev/experiment3/{sch}/55_4112/rp_{sch}.png", format="PNG")
