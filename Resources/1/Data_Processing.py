import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class COCOCustomDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the COCO dataset.
        :param image_dir: Path to the directory containing images.
        :param annotation_file: Path to the COCO-style annotation file.
        :param transform: Image transformations to apply.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)['annotations']

    def __len__(self):
        # Return the total number of samples
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get an image and its label by index.
        :param idx: Index of the sample.
        :return: Processed image tensor and label.
        """
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation['file_name'])
        label = annotation['category_id']  # Extract the label (class ID)

        # Open the image and apply transformations
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 for AlexNet
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load the training and testing datasets
train_dataset = COCOCustomDataset(
    image_dir='D:\Combined_COCO_Dataset',
    annotation_file='D:\Combined_COCO_Dataset\annotations.json',
    transform=transform
)

test_dataset = COCOCustomDataset(
    image_dir='D:\Combined_COCO_Dataset',
    annotation_file='D:\Combined_COCO_Dataset/annotations.json',
    transform=transform
)

# Create data loaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
