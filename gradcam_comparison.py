import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.models import resnet18
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ModifiedResnetAttention import ModifiedResNetAttention
from ModifiedResnetLayered import ModifiedResNetLayered
'''
GradCam Analysis provided by Claude.
'''

class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        
        return heatmap, target_class

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    return superimposed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 37


transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


models = []

# Model 1: ResNet18 Vanilla
model1 = resnet18(weights=None)
model1.name = 'Resnet18_vanilla'
model1.fc = nn.Linear(512, num_classes)
checkpoint1 = torch.load('checkpoints/Oxford Pets OLD/Resnet18_vanilla_oxford_best.pth', map_location=device, weights_only=False)
model1.load_state_dict(checkpoint1['model_state_dict'], strict=False)
model1 = model1.to(device)
model1.eval()
models.append(('ResNet18 Vanilla', model1, model1.layer4))

# Model 2: ResNet with Attention
model2 = ModifiedResNetAttention(num_classes=num_classes)
model2.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
checkpoint2 = torch.load('checkpoints/Oxford Pets OLD/ModifiedResNetAttention_oxford_best.pth', map_location=device, weights_only=False)
model2.load_state_dict(checkpoint2['model_state_dict'], strict=False)
model2 = model2.to(device)
model2.eval()
models.append(('ResNet Attention', model2, model2.layer4))

# Model 3: Split Architecture - Multiple layers
model3 = ModifiedResNetLayered(num_classes=num_classes)
model3.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
checkpoint3 = torch.load('checkpoints/Oxford Pets OLD/ModifiedResNetLayered_oxford_best.pth', map_location=device, weights_only=False)
model3.load_state_dict(checkpoint3['model_state_dict'], strict=False)
model3 = model3.to(device)
model3.eval()

# Add split architecture with multiple layers
models.append(('Split - Conv Branch', model3, model3.Convolution_layer3))
models.append(('Split - Attention Branch', model3, model3.attention_layer3))
models.append(('Split - Layer4', model3, model3.layer4))


test_dataset = torchvision.datasets.OxfordIIITPet(
    root="data/",
    split='test',
    transform=None,
    download=False
)
class_names = test_dataset.classes

IMAGE_INDEX = 10  # Change this to test different images
original_img, actual_class_idx = test_dataset[IMAGE_INDEX]
actual_class_name = class_names[actual_class_idx]

print(f"Processing image index: {IMAGE_INDEX}")
print(f"Actual class: {actual_class_name} (index: {actual_class_idx})")
print("=" * 60)

original_np = np.array(original_img)
input_tensor = transform(original_img).unsqueeze(0).to(device)

results = []
for name, model, layer in models:
    gradcam = GradCAM(model, layer)
    heatmap, pred_class = gradcam.generate_cam(input_tensor.clone())
    overlay = overlay_heatmap(original_np, heatmap)
    pred_class_idx = pred_class.item()
    pred_class_name = class_names[pred_class_idx]
    is_correct = pred_class_idx == actual_class_idx
    results.append((name, heatmap, overlay, pred_class_idx, pred_class_name, is_correct))
    
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    print(f"{name:30s} | Predicted: {pred_class_name:20s} (idx: {pred_class_idx:2d}) | {status}")

print("=" * 60)

# Visualize all results
n_models = len(results)
fig, axes = plt.subplots(n_models, 3, figsize=(15, 5*n_models))

if n_models == 1:
    axes = axes.reshape(1, -1)

for i, (name, heatmap, overlay, pred_idx, pred_name, is_correct) in enumerate(results):
    status_symbol = "✓" if is_correct else "✗"
    status_color = "green" if is_correct else "red"
    
    axes[i, 0].imshow(original_np)
    axes[i, 0].set_title(f'{name}\nActual: {actual_class_name}', fontsize=10)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(heatmap, cmap='jet')
    axes[i, 1].set_title('Grad-CAM Heatmap', fontsize=10)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[i, 2].set_title(f'{status_symbol} Predicted: {pred_name}', 
                         fontsize=10, color=status_color, weight='bold')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig(f'GradCam/GradCam_image_{IMAGE_INDEX}_class_{actual_class_name}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as 'gradcam_comparison_all.png'")
