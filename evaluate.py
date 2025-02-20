import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from algorithms.burn_ganext50 import BuRnGANeXt50

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BuRnGANeXt50(num_classes=3).to(device)
model.load_state_dict(torch.load("models/trained_model.pth"))
model.eval()

# Load Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(root='data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluation Loop
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
