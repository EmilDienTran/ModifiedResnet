import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from ModifiedResnetAttention import ModifiedResNetAttention
from torchvision.models import resnet18, ResNet18_Weights
import torch.multiprocessing as mp
import numpy as np


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    transforms_train = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.AutoAugment(policy=v2.AutoAugment.CIFAR10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            [0.50707516, 0.48654887, 0.44091784],
            [0.26733429, 0.25643846, 0.27615047])
    ])

    transform_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            [0.50707516, 0.48654887, 0.44091784],
            [0.26733429, 0.25643846, 0.27615047])
    ])

    train_dataset = torchvision.datasets.CIFAR100(root="data/", train=True, transform=transforms_train)
    test_dataset = torchvision.datasets.CIFAR100(root="data/", train=False, transform=transform_test)



    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=128,
                                                shuffle=True,
                                                num_workers=8,
                                                pin_memory=True,
                                                persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    Resnet18_normal = False
    if Resnet18_normal == True:
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, 100),
    else:
        model = ModifiedResNetAttention()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"\nModel is on: {next(model.parameters()).device}")
    print(f"Model = {model}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 180], gamma=0.2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=400, eta_min=1e-5)
    num_epochs = 400


    def sigmoidfunc(x):
        return 1 / (1 + np.exp(-x))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += torch.sum(labels == predicted.data).item()

            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
                #print(f"Conv branch weight: {sigmoidfunc(model.fusion.gamma.item()):.3f} | Attention Branch Weight: {1 - sigmoidfunc(model.fusion.gamma.item()):.3f}")
                #print(f"Attention weights: {model.layer2.MultiHeadAttention.attention_output}")

                running_loss = 0.0

        scheduler.step()
        train_accuracy = 100 * train_correct / train_total

        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += torch.sum(labels == predicted.data).item()

        test_loss /= len(test_loader)
        test_accuracy = 100 * correct / len(test_dataset)
        print(f"Epoch [{epoch + 1}], Train Accuracy: {train_accuracy:.4f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%")


