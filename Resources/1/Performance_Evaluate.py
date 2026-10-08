import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the best model weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set model to evaluation mode

# Collect predictions and true labels
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        # Move data to GPU/CPU
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Print the classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=[
    'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7']))

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
    'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7'],
    yticklabels=['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7'])
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title('Confusion Matrix')
plt.show()
