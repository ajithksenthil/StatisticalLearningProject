import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection

# Define transformations
transforms = transforms.Compose([
    transforms.Resize((800, 800)),  # Resize images for the model
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Load Pascal VOC 2007 dataset
voc_trainset = VOCDetection(root='./data', year='2007', image_set='train', download=True, transform=transforms)
voc_testset = VOCDetection(root='./data', year='2007', image_set='val', download=True, transform=transforms)
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
import torchvision.models.detection as detection

# Define transformations
transforms = transforms.Compose([
    transforms.Resize((800, 800)),  # Resize images for the model
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Load Pascal VOC 2007 dataset
voc_trainset = VOCDetection(root='./data', year='2007', image_set='train', download=True, transform=transforms)
voc_testset = VOCDetection(root='./data', year='2007', image_set='val', download=True, transform=transforms)


# Load a pre-trained Faster R-CNN model
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the classifier to match the number of classes in Pascal VOC
# Pascal VOC has 20 classes, plus the background class
num_classes = 21  # 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


from torch.utils.data import DataLoader
from torch.optim import SGD

# DataLoader setup
train_loader = DataLoader(voc_trainset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Optimizer setup
optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {losses.item()}')
