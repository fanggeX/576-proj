import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define the transformation to resize images to 128x128
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class CalligraphyDataset(Dataset):
    def __init__(self, calligraphy_type, transform=None):
        self.calligraphy_type = calligraphy_type
        self.transform = transform
        self.image_paths = self.load_images()
    
    def load_images(self):
        # Update the path based on the calligraphy type
        calligraphy_folder = {
            "seal": "./calligraphy-dataset/images/seal",
            "clerical": "./calligraphy-dataset/images/clerical",
            "cursive": "./calligraphy-dataset/images/cursive",
            "semi_cursive": "./calligraphy-dataset/images/semi-cursive",
            "regular": "./calligraphy-dataset/images/regular",
        }

        if self.calligraphy_type == "all":
            image_paths = []
            for folder in calligraphy_folder.values():
                image_paths.extend([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".jpg")])
        else:
            image_folder = calligraphy_folder.get(self.calligraphy_type, "")
            if not image_folder:
                raise ValueError(f"Invalid calligraphy type: {self.calligraphy_type}")
            image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if
                           file.endswith(".jpg")]

        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def add_white_box(self, image):
        # Add a white box to simulate partial characters
        h, w = image.shape[:2]
        # box_size = min(h, w) // 4
        box_size = 20

        ##### for random boxes
        # start_h = np.random.randint(h//2 - 30, h//2 + 10)
        # start_w = np.random.randint(w // 2 - 30, w // 2 + 10)

        ##### for fixed central box
        start_h = h//2 - 10
        start_w = w // 2 - 10
        image[start_h:start_h + box_size, start_w:start_w + box_size] = 255.0  # Set to white
        return image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        image_original = np.copy(image)
        
        image_partial = self.add_white_box(image)

        # Apply the transformation
        if self.transform:
            image_original = self.transform(image_original)
            image_partial = self.transform(image_partial)

        # Make the images binary to simplify
        image_original_binary = 1 - (image_original >= 0.3).float()
        image_partial_binary = 1 - (image_partial >= 0.3).float()
        
        return {"original_image": image_original_binary, "partial_image": image_partial_binary}

# Specify the calligraphy type for the dataset
calligraphy_type = "semi_cursive"

# Create a dataset
dataset = CalligraphyDataset(calligraphy_type, transform=transform)

# Create a data loader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate through the data loader
for batch in dataloader:
    original_image = batch["original_image"]
    partial_image = batch["partial_image"]

