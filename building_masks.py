import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tqdm import tqdm

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the input and output folders
input_folder = 'S:/Projects/Data Science And ML/Partial_Invisibility/annotated_images'  # Folder path containing input images
output_folder = 'S:/Projects/Data Science And ML/Partial_Invisibility/b_Masks'  # Folder path to store the output binary masks
os.makedirs(output_folder, exist_ok=True)

# Preprocess function
transform = T.Compose([T.ToTensor()])

# Process each image in the input folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the color image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convert the color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocess the grayscale image
        image_tensor = transform(gray_image)
        image_tensor = image_tensor.unsqueeze(0)

        # Perform instance segmentation
        with torch.no_grad():
            predictions = model(image_tensor)

        # Get the predicted masks
        masks = predictions[0]['masks'].detach().cpu().numpy()

        # Create a binary mask for each detected instance
        binary_mask = np.zeros_like(gray_image, dtype=np.uint8)
        for i in range(masks.shape[0]):
            mask = masks[i, 0]
            binary_mask[mask > 0.5] = 255

        # Save the binary mask
        output_path = os.path.join(output_folder, f'{filename[:-4]}_mask.png')
        cv2.imwrite(output_path, binary_mask)
