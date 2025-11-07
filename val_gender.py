import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import timm
import os
from sklearn.metrics import accuracy_score
class GenderDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["gender"], dtype=torch.long)  
        return img, label


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
df = pd.read_excel("Img_test.xlsx", header=None)
df = df[[0, 2]]
df.columns = ["filename", "gender"]
test_ds = GenderDataset(df, "Image_test", transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)


model.load_state_dict(torch.load("gender_effb3.pth", map_location="cpu"))


model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to("cpu"), labels.to("cpu")
        preds = model(imgs)
        pred_labels = preds.argmax(1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels.numpy())

acc = accuracy_score(y_true, y_pred)
print(f" Accuracy: {acc:.3f}")