import os

# main_folder_path = "./checkpoints"
# for model in os.listdir(main_folder_path):
#     folder_path = os.path.join(main_folder_path, model)
#     for ckpt_file in os.listdir(folder_path):
#         if ckpt_file.endswith("50.ckpt"):
#             file_path = os.path.join(folder_path, ckpt_file)
#             os.remove(file_path)
#             print(f"Deleted file: {file_path}")

import csv
import numpy as np
import matplotlib.pyplot as plt

model_name1 = "metrics"

total_epochs = 10000
ylim = 0.0001
intervall = 200


epoch = []
iou = []
f1 = []
recall = []
precision = []
accuracy = []



file_path1 = f"./outputs/metrics/{model_name1}.csv"
with open(file_path1, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        if i >= 101 and i <= 110:
            epoch.append(str(row[0]))
            iou.append(float(row[7]))
            f1.append(float(row[6]))
            recall.append(float(row[5]))
            precision.append(float(row[4]))
            accuracy.append(float(row[3]))

print(len(accuracy))
print(sum(accuracy)/len(accuracy), sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1),sum(iou)/len(iou))
# max_iou_index = iou.index(max(iou))
# print(epoch[max_iou_index])
print(epoch)
timesteps = [int(i+1) for i in range(len(iou))]

# intervall_loss1 = [sum(loss1[i:i+intervall])/intervall for i in range(0, total_epochs, intervall)]
# intervall_loss2 = [sum(loss2[i:i+intervall])/intervall for i in range(0, total_epochs, intervall)]
# intervall_loss3 = [sum(loss3[i:i+intervall])/intervall for i in range(0, total_epochs, intervall)]
# intervall_epochs = [i+intervall/2 for i in range(0, total_epochs, intervall)]



plt.plot(timesteps, iou, color="red", label="IoU")
plt.xticks(timesteps)
plt.xlabel("Inference")
plt.ylabel("IoU")
plt.title(f"Experiment 4 - DDIM 3 Timesteps Sampling ")
plt.ylim(0.6, 0.7)
plt.legend()

plt.savefig(f"ddim3_512_sampling.png")      