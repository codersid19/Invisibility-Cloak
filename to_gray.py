
# # Load the image
# image_path = 'S:/Projects/Data Science And ML/Partial_Invisibility/foreground/frame_0.jpg'
# image = cv2.imread(image_path)
#
# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Display the grayscale image
# cv2.imshow('Grayscale Image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the color image
image_path = 'S:/Projects/Data Science And ML/Partial_Invisibility/foreground/frame_0.jpg'
image = cv2.imread(image_path)

# Convert the color image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess the grayscale image
transform = T.Compose([T.ToTensor()])
image_tensor = transform(gray_image)

# Add a batch dimension to the image tensor
image_tensor = image_tensor.unsqueeze(0)

# Perform instance segmentation
with torch.no_grad():
    predictions = model(image_tensor)

# Get the predicted masks, boxes, and labels
masks = predictions[0]['masks'].detach().cpu().numpy()
boxes = predictions[0]['boxes'].detach().cpu().numpy()
labels = predictions[0]['labels'].detach().cpu().numpy()

# Create a binary mask for each detected instance
binary_mask = np.zeros_like(gray_image, dtype=np.uint8)
for i in range(masks.shape[0]):
    mask = masks[i, 0]
    box = boxes[i]
    label = labels[i]

    # Convert the mask to binary
    binary_mask[mask > 0.5] = 255

    # Draw the bounding box and label on the image (optional)
    x, y, w, h = box
    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the binary mask and annotated image (optional)
cv2.imshow('Binary Mask', binary_mask)
cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
