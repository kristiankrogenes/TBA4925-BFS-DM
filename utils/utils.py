import os
import argparse

from torchvision import transforms
import numpy as np
from PIL import Image
import csv

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
    if not model in os.listdir("./inference"):
        raise ValueError("No existing inference on this model.")
    elif not f"{image_code}.png" in os.listdir(f"./inference/{model}"):
        raise ValueError("The model has not predicted this image code.") 
    
    image1 = Image.open(f"./inference/{model}/{image_code}.png")
    image2 = Image.open(f"./data/processed/predictions/val/{image_code}.png")
    image3 = Image.open(f"./data/original/ortophotos/{image_code}.png")
    image4 = Image.open(f"./data/original/labels/val/{image_code}.png")

    width, height = image1.size

    combined_image = Image.new('RGB', (2 * width, 2 * height))

    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width, 0))
    combined_image.paste(image3, (0, height))
    combined_image.paste(image4, (width, height))

    folder_path = os.path.join("./data/comparisons", model)

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


def check_parameters_for_training(epochs, size, batch_size, timesteps, parameterization, model_path, save_model):
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
    elif not model_path==None and not os.path.exists(model_path):
        raise ValueError("Model requested does not exist.")
    elif save_model in os.listdir("./checkpoints") and model_path is None:
        raise ValueError("Model already exists, choose another name.")
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters to compare images.")
    parser.add_argument('--model', type=str, help="Name of a trained model.")
    parser.add_argument('--image_code', type=str, help="Image code of a given image.")
    args = parser.parse_args()

    compare_inference_pred_label_orto(args.model, args.image_code)