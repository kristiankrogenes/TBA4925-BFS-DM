import matplotlib.pyplot as plt
import csv


epochs = []
loss = []

with open('./loss_metrics.csv', 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        epochs.append(int(row[0]))
        loss.append(float(row[1]))
    

# Plot the training loss
plt.plot(epochs, loss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.savefig('training_loss_plot.png')