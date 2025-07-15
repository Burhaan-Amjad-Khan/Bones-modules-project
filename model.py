import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Must match training architecture exactly
model = models.resnet50(weights="IMAGENET1K_V1")
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(2048, 2)
model.load_state_dict(torch.load("best_mura_model.pth", map_location=device))
model = model.to(device)
model.eval()

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return "Abnormal" if predicted.item() == 1 else "Normal"
