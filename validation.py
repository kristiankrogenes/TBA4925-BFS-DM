from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def calculate_metrics(preds, labels):

    average_accuracy = []
    average_precision = []
    average_recall = []
    average_f1 = []
    average_iou = []

    for index, (pred, label) in enumerate(zip(preds, labels)):

        pred[pred > 0] = 1

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

    print(f"Metrics for {preds.__len__()} predictions ===================")
    print("Accuracy:", sum(average_accuracy) / len(average_accuracy))
    print("Precision:", sum(average_precision) / len(average_precision))
    print("Recall:", sum(average_recall) / len(average_recall))
    print("F1:", sum(average_f1) / len(average_f1))
    print("IoU:", sum(average_iou) / len(average_iou))