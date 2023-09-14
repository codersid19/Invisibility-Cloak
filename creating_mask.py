import os
import torch
import torchvision
import cv2
import numpy as np
from tqdm import tqdm

# Load the pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the input and output folders
input_folder = 'S:/Projects/Data Science And ML/Partial_Invisibility/foreground'  # Folder path containing input images
output_folder = 'S:/Projects/Data Science And ML/Partial_Invisibility/annotated_images'  # Folder path to store the output images
os.makedirs(output_folder, exist_ok=True)

# Preprocess function
transform = torchvision.transforms.ToTensor()

# Process each image in the input folder
for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the input image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Preprocess the image
        image_tensor = transform(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        # Make predictions on the image
        with torch.no_grad():
            predictions = model(image_tensor)

        # Extract masks and boxes from the predictions
        masks = predictions[0]['masks'].detach().cpu().numpy()
        boxes = predictions[0]['boxes'].detach().cpu().numpy()

        # Iterate over each mask and draw it on the image
        for i in range(masks.shape[0]):
            mask = masks[i, 0]
            box = boxes[i]

            # Threshold the mask to convert it into binary
            thresholded_mask = (mask > 0.5).astype(np.uint8) * 255

            # Draw the mask on the image
            contours, _ = cv2.findContours(thresholded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

            # # Draw the bounding box
            # x, y, w, h = box
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the resulting image with annotations
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
