import matplotlib.pyplot as plt

# replace these with your actual numbers
epochs = [1, 2, 3, 4, 5]
train_losses = [0.0527, 0.0248, 0.0223, 0.0190, 0.0100]
val_losses   = [0.0351, 0.0125, 0.0384, 0.0140, 0.0033]

plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses, marker='o', label='Train Loss')
plt.plot(epochs, val_losses,   marker='o', label='Validation Loss')
plt.title('Fine-tuning Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.show()