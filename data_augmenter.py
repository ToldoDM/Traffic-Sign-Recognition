import os
import cv2
import numpy as np
import albumentations as A


class DataAugmenter:
    def __init__(self, dataset_path):
        self.dataset_images = []
        self.dataset_path = dataset_path

    def load_images(self):
        for class_folder in os.listdir(os.path.join(self.dataset_path, "original")):
            class_folder_path = os.path.join(self.dataset_path, "original", class_folder)
            for filename in os.listdir(class_folder_path):
                file_path = os.path.join(class_folder_path, filename)
                if filename.endswith('.ppm'):  # Filter by image file extensions
                    print(file_path)
                    # img = cv2.imread(image_path)
                    # if img is not None:
                    #     self.dataset_images.append([img, ])
                    # else:
                    #     print(f"Failed to load image: {image_path}")


da = DataAugmenter("dataset/GTSRB/training")
da.load_images()