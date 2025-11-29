import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch.nn as nn
import torch.optim as optim
from ModifiedResnetAttention import ModifiedResNetAttention
from ModifiedResnetLayered import ModifiedResNetLayered
from torchvision.models import resnet18
import torch.multiprocessing as mp
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2, AutoAugmentPolicy
import torch
import torchvision
from thop import profile, clever_format
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix



def sigmoidfunc(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    models = []
    dataset = 'oxford'

    if dataset == 'cifar100':
        transforms_train = v2.Compose([
            v2.RandomCrop(size=(32, 32), padding=4),
            v2.RandomHorizontalFlip(p=0.25),
            v2.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                [0.50707516, 0.48654887, 0.44091784],
                [0.26733429, 0.25643846, 0.27615047]),
            v2.RandomErasing(p=0.25),
        ])

        transform_test = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                [0.50707516, 0.48654887, 0.44091784],
                [0.26733429, 0.25643846, 0.27615047])
        ])

        train_dataset = torchvision.datasets.CIFAR100(root="data/", train=True, transform=transforms_train,
                                                           download=True)
        test_dataset = torchvision.datasets.CIFAR100(root="data/", train=False, transform=transform_test,
                                                           download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   num_workers=12,
                                                   pin_memory=True,
                                                   persistent_workers=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=64,
                                                  shuffle=False)
    if dataset=='oxford':
        transforms_train = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.RandomHorizontalFlip(p=0.25),
            v2.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            v2.RandomErasing(p=0.25),
        ])

        transform_test = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        train_dataset = torchvision.datasets.OxfordIIITPet(root="data/",
                                                           split='trainval',
                                                           transform=transforms_train,
                                                           download=True
                                                           )
        test_dataset = torchvision.datasets.OxfordIIITPet(root="data/",
                                                          split='test',
                                                          transform=transform_test,
                                                          download=True
                                                          )
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=16,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   persistent_workers=True
                                                   )
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=16,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                  )



    num_classes = len(train_dataset.classes)
    print(f"Dataset has {len(train_dataset)} samples")

    model = resnet18(weights=None, progress=True)
    model.name = 'Resnet18_vanilla'
    if dataset == 'cifar100':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(512, num_classes)
    models.append(model)


    #Resnet without Branches - Attention
    model = ModifiedResNetAttention(num_classes=num_classes)
    if dataset == 'oxford':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    models.append(model)


    #Resnet with two Branches - Attention
    model = ModifiedResNetLayered(num_classes=num_classes)
    if dataset == 'oxford':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    models.append(model)


    for run_count in range(1):
        for model in models:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Training setup
            num_epochs = 200
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.SGD(model.parameters(), lr=0.07, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-5)

            '''
            Print structure provided via CLAUDE - all print code by Emil Dien Tran
            '''
            # ============================================
            # SYSTEM INFO
            # ============================================
            print(f"\n{'=' * 50}")
            print(f"CUDA DEVICE INFO")
            print(f"{'=' * 50}")
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"Model device: {next(model.parameters()).device}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

            # ============================================
            # MODEL ARCHITECTURE
            # ============================================
            print(f"\n{'=' * 50}")
            print(f"MODEL: {model.name}")
            print(f"{'=' * 50}")
            print(model)

            # ============================================
            # MODEL STATISTICS
            # ============================================
            print(f"\n{'=' * 50}")
            print(f"MODEL STATISTICS")
            print(f"{'=' * 50}")

            # Parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

            # FLOPs using thop
            if dataset == 'cifar100':
                input_tensor = torch.randn(1, 3, 32, 32).to(device)
            elif dataset == 'oxford':
                input_tensor = torch.randn(1, 3, 224, 224).to(device)
            flops, params = profile(model, inputs=(input_tensor,), verbose=False)
            flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
            print(f"MACs: {flops:,} ({flops / 1e9:.3f} GMACs)")
            print(f"FLOPs: {flops * 2:,} ({flops * 2 / 1e9:.3f} GFLOPs)")

            # ============================================
            # TRAINING CONFIGURATION
            # ============================================
            print(f"\n{'=' * 50}")
            print(f"TRAINING CONFIG")
            print(f"{'=' * 50}")
            print(f"Epochs: {num_epochs}")
            print(f"Optimizer: SGD (lr=0.07, momentum=0.9, weight_decay=5e-4)")
            print(f"Scheduler: CosineAnnealingLR (eta_min=1e-5)")
            print(f"Loss: CrossEntropyLoss")

            print(f"\n{'=' * 50}")
            print(f"STARTING TRAINING")
            print(f"{'=' * 50}\n")


            start_time = time.time()
            writer = SummaryWriter(f'runs/OXFORD/{model.name}_Epochs_{num_epochs}_Run_{run_count}_{dataset}_{time.strftime("%Y%m%d_%H%M%S")}')
            os.makedirs('checkpoints', exist_ok=True)
            best_val_acc = 0.0

            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                train_correct = 0
                train_total = 0

                epoch_start = time.time()
                train_start = time.time()
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

                train_time = time.time() - train_start
                scheduler.step()
                train_accuracy = 100 * train_correct / train_total

                model.eval()
                test_loss = 0.0
                correct = 0
                eval_start = time.time()

                all_targets = []
                all_predicted = []

                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        correct += torch.sum(labels == predicted.data).item()
                        all_targets.extend(labels.cpu().numpy())
                        all_predicted.extend(predicted.cpu().numpy())

                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_targets, all_predicted, average='weighted'
                )

                epoch_time = time.time() - epoch_start
                test_loss /= len(test_loader)
                test_accuracy = 100 * correct / len(test_dataset)
                eval_time = time.time() - eval_start


                # Model Saving for GradCam analysis - provided by LLM (Claude)
                if test_accuracy > best_val_acc:
                    os.makedirs(f'checkpoints/Oxford Pets {run_count}', exist_ok=True)
                    best_val_acc = test_accuracy
                    checkpoint_path = f'checkpoints/Oxford Pets {run_count}/{model.name}_{dataset}_best.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'test_accuracy': test_accuracy,
                        'train_accuracy': train_accuracy,
                    }, checkpoint_path)

                    cm = confusion_matrix(all_targets, all_predicted)
                    print("\nConfusion Matrix")
                    print(cm)
                    np.save(f'checkpoints/Oxford Pets {run_count}/{model.name}_{dataset}_confusion_matrix.npy', cm)
                    print(f"✓ Saved best model: {test_accuracy:.2f}%")


                # === TENSORBOARD LOGGING ===
                writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                writer.add_scalar('Accuracy/test', test_accuracy, epoch)
                writer.add_scalar('precision/test', precision, epoch)
                writer.add_scalar('recall/test', recall, epoch)
                writer.add_scalar('F1-score/test', f1, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalar('Time/epoch', epoch_time, epoch)
                writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

                if model.name == 'ModifiedResNetLayered':
                    writer.add_scalar('Fusion/conv_weight', sigmoidfunc(model.fusion.gamma.item()), epoch)
                    writer.add_scalar('Fusion/attention_weight', 1 - sigmoidfunc(model.fusion.gamma.item()), epoch)
                    writer.add_scalar('Fusion/gamma_raw', model.fusion.gamma.item(), epoch)

                print(f"Epoch [{epoch + 1}] | Train Accuracy: {train_accuracy:.4f}% |"
                      f" Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}% |"
                      f"Epoch Time: {epoch_time:.1f}s | Evalualtion Time: {eval_time:.3f}s | Training Time: {train_time:.3f}s")
            total_time = time.time() - start_time
            print(f"\nTotal training time: {total_time / 60:.1f} minutes")

            writer.close()
            del criterion
            del optimizer
            del scheduler
            model.cpu()
            del model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()