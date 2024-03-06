import torch
from src.models import ViTForObjectDetection
from src.dataset import VOCDatset as PascalVOCDataset
# Assume VOC2007Dataset is implemented
# dataset = VOC2007Dataset('path/to/voc2007/', transforms=...)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

from torch.utils.data import DataLoader

# Adjust paths and parameters according to your setup
train_dataset = PascalVOCDataset(root='./VOCdevkit', year='2007', image_set='trainval', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = ViTForObjectDetection(num_classes=20)  # 20 classes in Pascal VOC
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForObjectDetection(num_classes=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {losses.item()}")


# comments added by GPT, plus models suggested and code assisted with GPT

