import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.segmentation import deeplabv3_resnet50

# Define your custom dataset for loading the labeled data
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Define the paths to your labeled dataset
image_dir = 'path_to_images_folder'
mask_dir = 'path_to_masks_folder'

# Define hyperparameters
num_classes = 2
learning_rate = 0.001
batch_size = 16
num_epochs = 10

# Create a list of image and mask file paths
image_paths = [os.path.join(image_dir, img_file) for img_file in os.listdir(image_dir)]
mask_paths = [os.path.join(mask_dir, mask_file) for mask_file in os.listdir(mask_dir)]

# Split the dataset into training and validation sets
train_size = int(0.8 * len(image_paths))
train_image_paths = image_paths[:train_size]
train_mask_paths = mask_paths[:train_size]
val_image_paths = image_paths[train_size:]
val_mask_paths = mask_paths[train_size:]

# Define the data transformations
transform = ToTensor()

# Create instances of the custom dataset for training and validation
train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, transform=transform)
val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform=transform)

# Create data loaders for efficient data handling during training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the segmentation model
model = deeplabv3_resnet50(pretrained=True)
model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)['out']
            loss = criterion(outputs, masks.long())

            val_loss += loss.item()

        val_loss /= len(val_loader)

    # Print the epoch and the corresponding training and validation losses
    print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'path_to_save_model')
