import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET

class dataset_p(Dataset):
    def __init__(self, S=7, B=2, C=20, img_shape=(3, 448, 448), transform=None):
        super().__init__()
        self.img_shape = img_shape
        self.S = S
        self.B = B
        self.C = C
        
        # Initialize Pascal 2007 Dataset
        self.voc_dataset = VOCDetection(
            root='.',
            year='2007',
            image_set='train',
            download=False,
            transform=None
        )

        self.classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

    @staticmethod
    def convert_bbox_to_yolo(
            size: tuple[float, float], # w,h
            box: tuple[float, float, float, float], # xmin, ymin, xmax, ymax
    ) -> tuple[float, float, float, float]: #x_center, y_center, w, h
        
        scale_width = 1.0 / size[0]
        scale_height = 1.0 / size[1]

        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]

        # Converting box coords from absolute to relative
        rel_center_x = center_x * scale_width
        rel_center_y = center_y * scale_height
        rel_width = box_width * scale_width
        rel_height = box_height * scale_height

        return (rel_center_x, rel_center_y, rel_width, rel_height) # x, y, w, h
    
    def parse_xml_annotation(self, annotation):
        objects = []
        
        # Get image size
        size = annotation['annotation']['size']
        img_width = int(size['width'])
        img_height = int(size['height'])
        
        # Handle both single object and multiple objects cases
        obj_list = annotation['annotation']['object']
        if not isinstance(obj_list, list):
            obj_list = [obj_list]
        
        for obj in obj_list:
            class_name = obj['name']
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                
                # Extract bounding box coordinates
                bbox = obj['bndbox']
                xmin = float(bbox['xmin'])
                ymin = float(bbox['ymin'])
                xmax = float(bbox['xmax'])
                ymax = float(bbox['ymax'])
                
                # Convert to YOLO format (relative coordinates)
                yolo_bbox = self.convert_bbox_to_yolo(
                    (img_width, img_height),
                    (xmin, ymin, xmax, ymax)
                )
                
                objects.append({
                    'class_idx': class_idx,
                    'bbox': yolo_bbox
                })
        
        return objects, (img_width, img_height)
    
    def create_tensor(self, objects):
        # Initialize target tensor: S x S x (5*B + C)
        target = torch.zeros(self.S, self.S, 5 * self.B + self.C)
        
        for obj in objects:
            class_idx = obj['class_idx']
            x_center, y_center, width, height = obj['bbox']
            
            # Determine which grid cell this object belongs to
            grid_x = int(x_center * self.S)
            grid_y = int(y_center * self.S)
            
            # Ensure grid coordinates are within bounds
            grid_x = min(grid_x, self.S - 1)
            grid_y = min(grid_y, self.S - 1)
            
            # Calculate relative position within the grid cell
            rel_x = x_center * self.S - grid_x
            rel_y = y_center * self.S - grid_y
            
            # If this grid cell doesn't already have an object
            if target[grid_y, grid_x, self.C + 4] == 0:  # Check if confidence is 0
                # Set class probabilities
                target[grid_y, grid_x, 5 * self.B + class_idx] = 1.0
                # Set bounding box coordinates for first box
                target[grid_y, grid_x, 0] = rel_x
                target[grid_y, grid_x, 1] = rel_y
                target[grid_y, grid_x, 2] = width
                target[grid_y, grid_x, 3] = height
                target[grid_y, grid_x, 4] = 1.0  # Confidence
                
                # If we have multiple bounding boxes per cell, set the second box to be the same
                if self.B > 1:
                    target[grid_y, grid_x, 5] = rel_x
                    target[grid_y, grid_x, 6] = rel_y
                    target[grid_y, grid_x, 7] = width
                    target[grid_y, grid_x, 8] = height
                    target[grid_y, grid_x, 9] = 1.0  # Confidence for second box
        
        return target
    
    def __len__(self):
        return len(self.voc_dataset)
    
    def __getitem__(self, index):
        image, annotation = self.voc_dataset[index]
        objects, original_size = self.parse_xml_annotation(annotation)
        
        if self.transform:
            image = self.transform(image)
        
        target = self.create_tensor(objects)
        
        return image, target


# Testing
if __name__ == "__main__":
    
    # Test the dataset
    dataset = dataset_p(S=7, B=2, C=20)
    
    print(f"Dataset length: {len(dataset)}")
    
    image, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Target shape: {target.shape}")
    
    # Check if there are any objects in the target
    confidence_mask = target[:, :, 4] > 0
    num_objects = confidence_mask.sum().item()
    print(f"Number of objects detected: {num_objects}")
    
    # Statistics
    for i in range(5):
        image, target = dataset[i]
        confidence_mask = target[:, :, 4] > 0
        num_objects = confidence_mask.sum().item()
        print(f"Sample {i}: {num_objects} objects")

    if num_objects > 0:
        print("Grid cells with objects:")
        for i in range(7):
            for j in range(7):
                if target[i, j, 4] > 0:
                    class_probs = target[i, j, 10:]
                    class_idx = torch.argmax(class_probs).item()
                    print(f"  Cell ({i},{j}): Class {dataset.classes[class_idx]}")

"""
Output:

Dataset length: 2501
Image shape: torch.Size([3, 448, 448])
Target shape: torch.Size([7, 7, 30])
Number of objects detected: 1
Sample 0: 1 objects
Sample 1: 2 objects
Sample 2: 6 objects
Sample 3: 1 objects
Sample 4: 4 objects
Grid cells with objects:
  Cell (2,2): Class aeroplane
  Cell (3,3): Class aeroplane
  Cell (5,0): Class person
  Cell (5,2): Class person

"""

    