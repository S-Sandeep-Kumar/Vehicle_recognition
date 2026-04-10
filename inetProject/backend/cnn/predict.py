import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "models"))
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.txt")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")

with open(CLASSES_PATH, "r") as f:
    classes = f.read().splitlines()

# Load Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# 📌 FIXED: Removed random augmentation for consistent prediction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    return classes[predicted.item()], confidence.item()

if __name__ == "__main__":
    test_image = "test2.jpg"
    label, conf = predict_image(test_image)
    print(f"Prediction: {label} ({conf*100:.2f}%)")