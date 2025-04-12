from flask import Flask, render_template, request, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'animals'
MODEL_PATH = 'best_footprint.pth'
SAFE_ANIMALS = {'deer', 'horse', 'mouse', 'squirrel', 'racoon'}
CLASS_LABELS = ['bear', 'bobcat', 'deer', 'fox', 'horse', 'lion', 'mouse', 'racoon', 'squirrel', 'wolf']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_LABELS))
        )

    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    label, confidence, image_filename, bg_color, status = None, None, None, "#ffffff", None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(tensor)
                    probs = F.softmax(out, dim=1)
                    idx = torch.argmax(probs).item()
                    label = CLASS_LABELS[idx]
                    confidence = f"{probs[0][idx].item() * 100:.2f}%"
                    is_safe = label in SAFE_ANIMALS
                    status = "Safe" if is_safe else "Unsafe"
                    bg_color = "lightgreen" if is_safe else "#ff6961"
                    image_filename = f"{label}.jpg"

    return render_template("index.html", label=label, confidence=confidence,
                           image_filename=image_filename, bg_color=bg_color, status=status)

@app.route('/animals/<filename>')
def send_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
