import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import torch
import kornia.augmentation as KA 
from torch.utils.data import DataLoader

from dataset import BFSDataset
from model.BFSDiffusionModel import BFSDiffusionModel
from utils.utils import transform_model_output_to_image

import csv

def save_dict_to_csv(dictionary, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
        writer.writeheader()
        writer.writerow(dictionary)

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

    # if verbose:
    #     print(f"Metrics for {preds.__len__()} predictions ===================")
    #     print("Accuracy:", sum(average_accuracy) / max(1, len(average_accuracy)))
    #     print("Precision:", sum(average_precision) / max(1, len(average_precision)))
    #     print("Recall:", sum(average_recall) / max(1, len(average_recall)))
    #     print("F1:", sum(average_f1) / max(1, len(average_f1)))
    #     print("IoU:", sum(average_iou) / len(average_iou))

    return {"Accuracy": sum(average_accuracy) / len(average_accuracy),
            "Precision": sum(average_precision) / len(average_precision),
            "Recall": sum(average_recall) / len(average_recall),
            "F1": sum(average_f1) / len(average_f1),
            "IoU": sum(average_iou) / len(average_iou)}

def run_inference(model, dataset, meta_data, save_metrics=True, save_outputs=True):

    average_accuracy = []
    average_precision = []
    average_recall = []
    average_f1 = []
    average_iou = []

    for i, data in enumerate(dataset):

        label, pred = data

        generated_tensor = model(batch_input=pred.unsqueeze(0))
        generated_tensor = generated_tensor.detach().cpu()

        if save_outputs:
            generated_image = transform_model_output_to_image(generated_tensor[0])

            if not os.path.exists(f"./inference/{meta_data['model_name']}"):
                os.makedirs(f"./inference/{meta_data['model_name']}")

            generated_image.save(f"./inference/{meta_data['model_name']}/{dataset.label_files[i]}", format="PNG")

        if save_metrics:

            metrics = calculate_metrics((generated_tensor.add_(1)).div_(2).numpy(), (label.add_(1)).div_(2).numpy())

            average_accuracy.append(metrics["Accuracy"])
            average_precision.append(metrics["Precision"])
            average_recall.append(metrics["Recall"])
            average_f1.append(metrics["F1"])
            average_iou.append(metrics["IoU"])
            
        print("Inference", i)
        if i==4:
            break

    if save_metrics:
        final_metrics = {"Model": meta_data["model_name"], 
                        "Epochs": meta_data["epochs"],
                        "Accuracy": sum(average_accuracy) / len(average_accuracy),
                        "Precision": sum(average_precision) / len(average_precision),
                        "Recall": sum(average_recall) / len(average_recall),
                        "F1": sum(average_f1) / len(average_f1),
                        "IoU": sum(average_iou) / len(average_iou)}
        
        final_metrics_row = [final_metrics["Model"],
                            final_metrics["Epochs"],
                            final_metrics["Accuracy"],
                            final_metrics["Precision"],
                            final_metrics["Recall"],
                            final_metrics["F1"],
                            final_metrics["IoU"]]
        add_row_to_csv("./metrics.csv", final_metrics_row)


if __name__ == "__main__":

    # // ARGUMENT PARSER // ============================================================================
    parser = argparse.ArgumentParser(description="Run inference and validation with model.")
    parser.add_argument('--model', type=str, help="Which model to check.")
    args = parser.parse_args()
    # ==================================================================================================
    
    if args.model == "baseline":
        meta = {"model_name": "Baseline", "epochs": 10000}
        checkpoint_path = f"./checkpoints/BaselineLabel/UNet_128x128_bs10_t100_v1_e10000.ckpt"
        timesteps = 100
    elif args.model == "baseline2":
        meta = {"model_name": "Baseline2", "epochs": 100}
        checkpoint_path = f"./checkpoints/BaselineLabelPred2/UNet_256x256_bs5_t100_v2_e{meta['epochs']}.ckpt"
        timesteps = 100
    elif args.model == "baseline3":
        meta = {"model_name": "Baseline3", "epochs": 1000}
        checkpoint_path = f"./checkpoints/BaselineLabel2/UNet_256x256_bs10_t100_v2_e{meta['epochs']}.ckpt"
        timesteps = 100
    elif args.model == "outputloss":
        meta = {"model_name": "OutputLoss2", "epochs": 5750}
        checkpoint_path = f"./checkpoints/{meta['model_name']}/UNet_128x128_bs16_t100_e{meta['epochs']}.ckpt"
        timesteps = 100
    elif args.model == "prebaseoutput":
        meta = {"model_name": "PreBaseOutput", "epochs": 20000}
        checkpoint_path = f"./checkpoints/{meta['model_name']}/UNet_128x128_bs16_t100_e{meta['epochs']}.ckpt"
        timesteps = 1000
    elif args.model == "prebaseoutputextended":
        meta = {"model_name": "PreBaseOutputExtended", "epochs": 50}
        checkpoint_path = f"./checkpoints/{meta['model_name']}/UNet_128x128_bs16_t100_e{meta['epochs']}.ckpt"
        timesteps = 100
    else:
        raise ValueError("This model does not exist.")

    bfs_model = BFSDiffusionModel(num_timesteps=timesteps, checkpoint=checkpoint_path)

    T = [
        KA.Resize(size=(512, 512), antialias=True),
        # KA.RandomHorizontalFlip(),
        # KA.ToTensor(), # Scales data into [0,1]
        # KA.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]

    train_dataset = BFSDataset(
        root_dir="data",
        type="train",
        transforms=T
    )

    val_dataset = BFSDataset(
        root_dir="data",
        type="val",
        transforms=T
    )

    run_inference(bfs_model, val_dataset, meta_data=meta, save_metrics=True, save_outputs=True)

    """
        Validate Norkart data
    """
    if False:
        average_accuracy = []
        average_precision = []
        average_recall = []
        average_f1 = []
        average_iou = []

        for batch in val_dataset:
            label, pred = batch
            metrics = calculate_metrics(pred, label)

            average_accuracy.append(metrics["Accuracy"])
            average_precision.append(metrics["Precision"])
            average_recall.append(metrics["Recall"])
            average_f1.append(metrics["F1"])
            average_iou.append(metrics["IoU"])

        final_metrics = {"Model": "Norkart", 
                        "Epochs": "NaN",
                        "Accuracy": sum(average_accuracy) / len(average_accuracy),
                        "Precision": sum(average_precision) / len(average_precision),
                        "Recall": sum(average_recall) / len(average_recall),
                        "F1": sum(average_f1) / len(average_f1),
                        "IoU": sum(average_iou) / len(average_iou)}
            
        final_metrics_row = [final_metrics["Model"],
                            final_metrics["Epochs"],
                            final_metrics["Accuracy"],
                            final_metrics["Precision"],
                            final_metrics["Recall"],
                            final_metrics["F1"],
                            final_metrics["IoU"]]
        add_row_to_csv("./metrics.csv", final_metrics_row)

