import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET


class VOCDataset(Dataset):
    def __init__(self, root_dir, year="2012", image_set="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            year (string): Year of the VOC dataset.
            image_set (string): One of "train", "trainval", "val", or "test".
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "VOCdevkit", "VOC" + year, "JPEGImages")
        self.annotation_dir = os.path.join(root_dir, "VOCdevkit", "VOC" + year, "Annotations")

        self._image_set_index = self._load_image_set_index()
        self.classes = self._load_classes()

    def __len__(self):
        return len(self._image_set_index)

    def __getitem__(self, idx):
        img_id = self._image_set_index[idx]
        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        annotation_path = os.path.join(self.annotation_dir, img_id + ".xml")

        img = Image.open(img_path).convert("RGB")
        boxes, labels = self._load_annotation(annotation_path)

        if self.transform:
            img, boxes = self.transform(img, boxes)

        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return img, target

    def _load_image_set_index(self):
        image_set_path = os.path.join(self.root_dir, "VOCdevkit", "VOC" + self.year, "ImageSets", "Main", self.image_set + ".txt")
        with open(image_set_path) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def _load_classes(self):
        # VOC dataset class names
        return ["__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def _load_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.iter('object'):
            label = obj.find('name').text
            labels.append(self.classes.index(label))
            bndbox = obj.find('bndbox')
            box = [int(bndbox.find(tag).text) - 1 for tag in ["xmin", "ymin", "xmax", "ymax"]]
            boxes.append(box)
        return boxes, labels

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train=True):
    transforms = []
    transforms.append(transforms.Resize((800, 800)))
    if train:
        # Data augmentation for training
        transforms.append(transforms.RandomHorizontalFlip(0.5))
        transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    transforms.append(transforms.ToTensor())
    transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transforms)

def custom_transform(img, target, transform):
    """Apply transformations to both image and target."""
    img = transform(img)
    # Here, apply any necessary transformations to `target` as well, such as resizing bounding boxes
    return img, target


# Example usage
if __name__ == "__main__":
    dataset = VOCDataset(root_dir="/path/to/VOCdevkit", year="2007", image_set="trainval", transform=get_transform())
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    for images, targets in data_loader:
        print(images[0].shape, targets[0])



"""
Download and Prepare the Pascal VOC 2012 Dataset: Make sure you've downloaded the dataset from its official website or a trusted source. This dataset includes images and annotations (in XML format) that describe the objects within each image.

Dataset Class: Implement a custom PyTorch dataset class that loads the images and their corresponding annotations, converts the annotations to a tensor format suitable for object detection models, and performs any required preprocessing steps (e.g., resizing images, normalizing pixel values).

Data Loader: Use PyTorch's DataLoader to efficiently load the dataset in batches during training.

"""        
# comments added, plus some methods assisted with GPT