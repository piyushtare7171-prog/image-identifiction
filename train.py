import os
import matplotlib.pyplot as plt

from dataset import train_ds, test_ds
from model import create_model

os.makedirs("saved_model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

model = create_model()

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds
)

model.save("saved_model/cnn_model.keras")

# Accuracy plot
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.savefig("plots/accuracy.png")
plt.show()

# Loss plot
plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.savefig("plots/loss.png")
plt.show()
