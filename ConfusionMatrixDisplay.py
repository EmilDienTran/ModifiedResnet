import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import torchvision

cm_Resnet18 = np.load('checkpoints/CIFAR100 3/Resnet18_vanilla_cifar100_confusion_matrix.npy')
cm_ModifiedResnetAttention = np.load('checkpoints/CIFAR100 3/ModifiedResNetAttention_cifar100_confusion_matrix.npy')
cm_ModifiedResnetLayered = np.load('checkpoints/CIFAR100 3/ModifiedResNetLayered_cifar100_confusion_matrix.npy')

cm_Resnet18 = cm_Resnet18[:30, :30]
cm_ModifiedResnetAttention = cm_ModifiedResnetAttention[:30, :30]
cm_ModifiedResnetLayered = cm_ModifiedResnetLayered[:30, :30]

test_dataset = torchvision.datasets.CIFAR100(root="data/", train=True,
                                                  download=True)
class_names = test_dataset.classes[:30]

disp = ConfusionMatrixDisplay(confusion_matrix=cm_Resnet18, display_labels=class_names)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(cmap='Blues', include_values=True, ax=ax, xticks_rotation='vertical')
plt.title('ResNet18 CIFAR100 Confusion Matrix')
plt.tight_layout()
plt.savefig('ConfusionMatrixDisplay_Resnet18_CIFAR100.png')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_ModifiedResnetAttention, display_labels=class_names)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(cmap='Blues', include_values=True, ax=ax, xticks_rotation='vertical')
plt.title('ModifedResnetAttention CIFAR100 Confusion Matrix')
plt.tight_layout()
plt.savefig('ConfusionMatrixDisplay_ModifiedResnetAttention_CIFAR100.png')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_ModifiedResnetLayered, display_labels=class_names)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(cmap='Blues', include_values=True, ax=ax, xticks_rotation='vertical')
plt.title('ModifedResnetLayered CIFAR100 Confusion Matrix')
plt.tight_layout()
plt.savefig('ConfusionMatrixDisplay_ModifiedResnetLayered_CIFAR100.png')
plt.show()