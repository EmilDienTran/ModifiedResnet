from  torchvision import datasets

cifar_trainset = datasets.CIFAR100(root='./data', train=True, download=True  )
data = cifar_trainset.data / 255

mean = data.mean(axis = (0,1,2))
std = data.std(axis = (0,1,2))
print(f"CIFAR100| Mean : {mean}   STD: {std}")

