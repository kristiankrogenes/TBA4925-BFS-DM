import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import torch
import kornia.augmentation as KA 
from torch.utils.data import DataLoader

from dataset import BFSDataset
from model.BFSDiffusionModel import BFSDiffusionModel
from model.LatentBFSDM import LatentBFSDM
from utils.utils import transform_model_output_to_image, add_row_to_csv
import model.config
import model.configV2
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_metrics(preds, labels, verbose=False):

    average_accuracy = []
    average_precision = []
    average_recall = []
    average_f1 = []
    average_iou = []

    for index, (pred, label) in enumerate(zip(preds, labels)):

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        label[label > 0] = 1
        label[label <= 0] = 0

        # print("###", pred.shape, label.shape)

        # pred_scaled = (pred * 255).astype(np.uint8)
        # label_scaled = (label * 255).astype(np.uint8)
        # img1 = Image.fromarray(pred_scaled[0])
        # img1.save("./predx1.png")
        # img2 = Image.fromarray(label_scaled)
        # img2.save("./labelx1.png")

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

    return {"Accuracy": sum(average_accuracy) / len(average_accuracy),
            "Precision": sum(average_precision) / len(average_precision),
            "Recall": sum(average_recall) / len(average_recall),
            "F1": sum(average_f1) / len(average_f1),
            "IoU": sum(average_iou) / len(average_iou)}

def run_inference(model, dataset, meta_data, save_metrics=True, save_outputs=True, ts=None):

    average_accuracy = []
    average_precision = []
    average_recall = []
    average_f1 = []
    average_iou = []

    for i, data in enumerate(dataset):

        # label, pred, orto = data
        label, pred = data

        generated_tensor = model(batch_input=pred.unsqueeze(0), ts=ts)

            
        generated_tensor = generated_tensor.detach().cpu()

        if save_outputs:
            generated_image = transform_model_output_to_image(generated_tensor[0])

            if not os.path.exists(f"./outputs/inference/norkart/{meta_data['model_name']}_e{meta_data['epoch']}"):
                os.makedirs(f"./outputs/inference/norkart/{meta_data['model_name']}_e{meta_data['epoch']}")

            generated_image.save(f"./outputs/inference/norkart/{meta_data['model_name']}_e{meta_data['epoch']}/{dataset.label_files[i]}", format="PNG")

            # label_image = transform_model_output_to_image(label)
            # label_image.save(f"./outputs/inference/{meta_data['model_name']}_e{meta_data['epoch']}/label_{dataset.label_files[i]}", format="PNG")

            # pred_image = transform_model_output_to_image(pred)
            # pred_image.save(f"./outputs/inference/{meta_data['model_name']}/pred_{dataset.label_files[i]}", format="PNG")


        if save_metrics:
            x1, x2 = (generated_tensor.add_(1)).div_(2).numpy(), (label.add_(1)).div_(2).numpy()
            # print(x1.shape, x2.shape)
            # print(np.min(x1), np.max(x1), np.min(x2), np.max(x2))
            # print(np.count_nonzero(x1 == 0), np.count_nonzero(x2 == 0))
            metrics = calculate_metrics(x1, x2, verbose=False)

            average_accuracy.append(metrics["Accuracy"])
            average_precision.append(metrics["Precision"])
            average_recall.append(metrics["Recall"])
            average_f1.append(metrics["F1"])
            average_iou.append(metrics["IoU"])
            
        print("Inference", i)

        if i == 1:
            break


    if save_metrics:
        final_metrics = {"Model": meta_data["model_name"], 
                        "Epochs": meta_data["epoch"],
                        "ImageSize": meta_data["size"],
                        "Accuracy": sum(average_accuracy) / len(average_accuracy),
                        "Precision": sum(average_precision) / len(average_precision),
                        "Recall": sum(average_recall) / len(average_recall),
                        "F1": sum(average_f1) / len(average_f1),
                        "IoU": sum(average_iou) / len(average_iou)}
        
        final_metrics_row = [final_metrics["Model"],
                            final_metrics["Epochs"],
                            final_metrics["ImageSize"],
                            final_metrics["Accuracy"],
                            final_metrics["Precision"],
                            final_metrics["Recall"],
                            final_metrics["F1"],
                            final_metrics["IoU"]]
        add_row_to_csv("./outputs/metrics/metrics.csv", final_metrics_row)


if __name__ == "__main__":

    # // ARGUMENT PARSER // ============================================================================
    parser = argparse.ArgumentParser(description="Run inference and validation with model.")
    parser.add_argument('--model', type=str, help="Which model to check.")
    parser.add_argument('--epoch', type=str, help="What epoch to check.")
    args = parser.parse_args()
    # ==================================================================================================
    
    if True:
        cfg = None
        for config in model.configV2.configs:
            if args.model==config.model_name:
                cfg = config
        if cfg is None:
            raise ValueError("This config does not exist.")
        
        if not cfg is None:
            checkpoint_path = f"./checkpoints/{cfg.model_name}/UNet_{cfg.TARGET_SIZE[0]}x{cfg.TARGET_SIZE[1]}_bs{cfg.batch_size}_t{cfg.timesteps}_e{args.epoch}.ckpt"
            cfg.TARGET_SIZE = (512,512)
            meta = {"model_name": cfg.model_name, "epoch": args.epoch, "size": cfg.TARGET_SIZE[0], "batch_size": cfg.batch_size, "timesteps": cfg.timesteps, "parameterization": cfg.parameterization, "condition_type": cfg.condition_type}
            timesteps = cfg.timesteps
            print(meta["size"])
        else:
            raise ValueError("This model does not exist.")
        
        bfs_model = BFSDiffusionModel(batch_size=cfg.batch_size,
                            image_size=cfg.TARGET_SIZE,
                            schedule=cfg.schedule,
                            num_timesteps=timesteps,
                            condition_type=cfg.condition_type,
                            sampler="DDIM",
                            parameterization=cfg.parameterization,
                            checkpoint=checkpoint_path)
        print("#", cfg.TARGET_SIZE)
        T=[
            # KA.RandomCrop((2*64,2*64)),
            KA.Resize(cfg.TARGET_SIZE, antialias=True),
            # KA.Resize((512,512), antialias=True),
            # KA.RandomVerticalFlip()
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
    print(len(val_dataset))

    # for i in range(0, 10):
    run_inference(bfs_model, val_dataset, meta_data=meta, save_metrics=True, save_outputs=True, ts=100)
    # run_inference(bfs_model, val_dataset, meta_data=meta, save_metrics=True, save_outputs=True)

    """
        Validate Norkart data
    """
    # T=[
    #     KA.Resize((512,512), antialias=True),
    # ]
    
    # val_dataset = BFSDataset(
    #         root_dir="data",
    #         type="val",
    #         transforms=T
    #     )
    if False:
        average_accuracy = []
        average_precision = []
        average_recall = []
        average_f1 = []
        average_iou = []

        # for batch in val_dataset:
        labels = os.listdir("./data/norkart/final/labels/")
        for pred in os.listdir("./data/NorkartSOTA/png/"):
            pred_array = np.asarray(Image.open("./data/NorkartSOTA/png/"+pred))
            pred_array = pred_array / 255
            label_array = np.asarray(Image.open("./data/norkart/final/labels/"+pred[:7]+".png"))

            pred_image_array_writable = pred_array.copy()
            pred_image_tensor = torch.FloatTensor(pred_image_array_writable)
            label_image_array_writable = label_array.copy()
            label_image_tensor = torch.FloatTensor(label_image_array_writable)
            if np.max(label_array) > 0:
                # print(pred)
                # print(pred_array.shape, np.min(pred_array), np.max(pred_array))
                # print(label_array.shape, np.min(label_array), np.max(label_array))
                # pred = np.expand_dims(pred_array, axis=0)
                # label = np.expand_dims(label_array, axis=0)
                pred = pred_image_tensor.unsqueeze(0)
                label = label_image_tensor.unsqueeze(0)
                # print(pred.shape, label.shape)
                # label, pred, ortho = batch
                metrics = calculate_metrics(pred, label, verbose=True)

                average_accuracy.append(metrics["Accuracy"])
                average_precision.append(metrics["Precision"])
                average_recall.append(metrics["Recall"])
                average_f1.append(metrics["F1"])
                average_iou.append(metrics["IoU"])

        final_metrics = {"Model": "Norkart",
                        "Epochs": "NaN",
                        # "ImageSize": cfg.TARGET_SIZE[0],
                        "ImageSize": 512,
                        "Accuracy": sum(average_accuracy) / len(average_accuracy),
                        "Precision": sum(average_precision) / len(average_precision),
                        "Recall": sum(average_recall) / len(average_recall),
                        "F1": sum(average_f1) / len(average_f1),
                        "IoU": sum(average_iou) / len(average_iou)}
            
        final_metrics_row = [final_metrics["Model"],
                            final_metrics["Epochs"],
                            final_metrics["ImageSize"],
                            final_metrics["Accuracy"],
                            final_metrics["Precision"],
                            final_metrics["Recall"],
                            final_metrics["F1"],
                            final_metrics["IoU"]]
        add_row_to_csv("./outputs/metrics/metrics.csv", final_metrics_row)

