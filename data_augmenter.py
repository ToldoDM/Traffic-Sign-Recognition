import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import random


def save_as_ppm(file_path, image):
    # Ensure the image is in uint8 format
    image = np.uint8(image)

    # Extract image dimensions
    height, width, channels = image.shape

    # Open the file for writing
    with open(file_path, 'wb') as f:
        # Write the PPM header
        header = "P6\n{} {}\n255\n".format(width, height)
        f.write(header.encode())

        # Write the image data
        f.write(image.tobytes())

class DataAugmenter:
    """
    A class for loading and augmenting dataset images stored in a specific nested directory structure.

    This class expects the dataset directory to follow a specific format:
    dataset/GTSRB/train/$class_num$
    where "$class_num$" represents different class directories containing images.
    """

    def __init__(self, dataset_path):
        """
        Initializes the DataAugmenter with a specified dataset path.

        Args:
            dataset_path (str): The root path where the dataset is stored, expected to be in the format
                                'GTSRB/merged/$class_num$', so the root dataset_path expected is 'GTSRB'.
        """
        self.loaded_images = []
        self.dataset_path = dataset_path
        self.folder_to_load_path = None
        self.classes = []
        self.delete_source=False

    def load_images(self, classes_to_load=None, folder_to_load='train'):
        """
        Loads '.ppm' images from a specified class subdirectory within the dataset path.
        It appends each image along with its metadata (class and filename) to the dataset_images list.
        This method navigates through the structured folder as: 'folder/$class_num$'.

        Args:
            classes_to_load (list): list of specific class names to load.
            folder_to_load (str): the folder type to load. Accepted values are 'train' and 'test'.
        """
        self.loaded_images = []
        self.folder_to_load_path = os.path.join(self.dataset_path, folder_to_load)
        self.classes = classes_to_load if classes_to_load is not None else os.listdir(self.folder_to_load_path)
        print("Classes found: ", *self.classes, sep=', ')
        for class_folder in tqdm(self.classes, desc='Loading classes'):
            class_folder_path = os.path.join(self.folder_to_load_path, class_folder)
            if not os.path.exists(class_folder_path):
                print('Class folder {} does not exist.'.format(class_folder))
            else:
                for filename in os.listdir(class_folder_path):
                    file_path = os.path.join(class_folder_path, filename)
                    if filename.endswith('.ppm'):  # Filter by image file extensions
                        img = cv2.imread(file_path)
                        # OpenCV reads images in BGR format, convert it to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if img_rgb is not None:
                            self.loaded_images.append(
                                [img_rgb,
                                 {'class': class_folder,
                                  'filename': os.path.splitext(filename)[0]}
                                 ])
                        else:
                            print(f"Failed to load image: {file_path}")
                self.loaded_images.sort(key=lambda x: (x[1]['class'], x[1]['filename']))

    def add_weather_effects(self, prob_per_class=0):
        """
                Substitute an image with a weather augmented one following a specified probability over the class

                Args:
                    prob_per_class (int): The probability of the image being processed per class.

                Side Effects:
                    - Substitute an image with a weather augmented one.
        """
        for class_label in tqdm(self.classes, desc='Add weather effects with probability '+str(prob_per_class)):
            # Filter images belonging to the current class
            class_images = [(img, metadata) for img, metadata in self.loaded_images if
                            metadata['class'] == class_label]

            # Randomly choose weather augmentation options with equal probability
            augmentation_options = [
                'rain', 'spatter', 'sun_flare', 'fog', 'shadow'
            ]

            # Calculate the number of images needed to reach the desired total
            for image,metadata in class_images:
                selected_filter = random.choice(augmentation_options)
                # Apply the augmentation with probability defined and delete the source image
                if (random.random() < prob_per_class):
                    self.apply_augmentation(image, selected_filter, metadata, delete_source=True)

    def balance_samples_per_class(self, num_of_total_images=0):
        """
        Augments images in the dataset to reach a specified total number of images per class.

        Args:
            num_of_total_images (int): The desired total number of images per class after augmentation.

        Side Effects:
            - Augments images in the dataset to meet the desired total number per class.
        """
        # Iterate over each class in the dataset
        for class_label in tqdm(self.classes, desc='Augmenting images'):
            # Filter images belonging to the current class
            class_images = [(img, metadata) for img, metadata in self.loaded_images if
                            metadata['class'] == class_label]

            # Calculate the number of images needed to reach the desired total
            num_existing_images = len(class_images)
            num_images_to_augment = num_of_total_images - num_existing_images

            # If the desired total is already reached or exceeded, skip augmentation
            if num_images_to_augment <= 0:
                print(f'Class {class_label} already has {num_existing_images} '
                      f'images that is higher than the requested total of {num_images_to_augment} images.')
                continue

            # Randomly choose augmentation options with equal probability
            augmentation_options = [
                'rain', 'spatter', 'zoom_blur', 'sun_flare', 'ringing_overshoot',
                'perspective', 'motion_blur', 'fog', 'ISO_noise', 'gamma', 'shadow'
            ]

            # Iterate until the desired number of images is augmented for the class
            while num_images_to_augment > 0:
                # Randomly select an augmentation option and image
                selected_option = random.choice(augmentation_options)
                selected_image, metadata = random.choice(class_images)
                metadata['iteration'] = num_images_to_augment

                # Apply the selected augmentation option
                self.apply_augmentation(selected_image, selected_option, metadata)

                # Decrement the number of images left to augment
                num_images_to_augment -= 1

    def apply_augmentation(self, sign_image, selected_option, metadata, delete_source=False):
        self.delete_source=delete_source

        # Upscale image for future use
        transform_image = A.Compose([
            A.SmallestMaxSize(p=1.0, max_size=128, interpolation=1)
        ])
        resized_image = transform_image(image=sign_image)['image']

        if selected_option == 'rain':
            self.apply_rain_effect(sign_image, resized_image, metadata)
        elif selected_option == 'spatter':
            self.apply_spatter_effect(sign_image, resized_image, metadata)
        elif selected_option == 'zoom_blur':
            self.apply_zoom_blur_effect(sign_image, resized_image, metadata)
        elif selected_option == 'sun_flare':
            self.apply_sun_flare_effect(sign_image, resized_image, metadata)
        elif selected_option == 'ringing_overshoot':
            self.apply_ringing_overshoot_effect(sign_image, resized_image, metadata)
        elif selected_option == 'perspective':
            self.apply_perspective_effect(sign_image, resized_image, metadata)
        elif selected_option == 'motion_blur':
            self.apply_motion_blur_effect(sign_image, resized_image, metadata)
        elif selected_option == 'fog':
            self.apply_fog_effect(sign_image, resized_image, metadata)
        elif selected_option == 'ISO_noise':
            self.apply_ISO_noise_effect(sign_image, resized_image, metadata)
        elif selected_option == 'gamma':
            self.apply_gamma_effect(sign_image, resized_image, metadata)
        elif selected_option == 'shadow':
            self.apply_shadow_effect(sign_image, resized_image, metadata)

    def apply_rain_effect(self, sign_image, resized_image, metadata):
        sign_img_height = np.shape(sign_image)[0]
        sign_img_width = np.shape(sign_image)[1]
        brightness_coefficient = random.choice([0.7, 0.5, 0.4, 0.3])
        drop_width = random.choice([1, 2])
        blur_value = random.choice([5, 7, 10])
        rain_type = random.choice(['drizzle', 'heavy', 'torrential'])
        transform_image = A.Compose([
            A.RandomRain(brightness_coefficient=brightness_coefficient,
                         drop_width=drop_width, blur_value=blur_value,
                         p=1, rain_type=rain_type),
            A.Resize(p=1.0, height=sign_img_height, width=sign_img_width, interpolation=1)
        ])
        transformed = transform_image(image=resized_image)
        metadata['type'] = 'rain'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_spatter_effect(self, sign_image, resized_image, metadata):
        spatter_mode = random.choice(['rain', 'mud'])
        transform_image = A.Compose([
            A.Spatter(p=1.0, mode=spatter_mode)
        ])
        transformed = transform_image(image=sign_image)
        metadata['type'] = 'spatter'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_zoom_blur_effect(self, sign_image, resized_image, metadata):
        transform_image = A.Compose([
            A.ZoomBlur(p=1.0, max_factor=(1.4, 2), step_factor=(0.03, 0.03))
        ])
        transformed = transform_image(image=sign_image)
        metadata['type'] = 'zoom_blur'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_sun_flare_effect(self, sign_image, resized_image, metadata):
        sign_img_height = np.shape(sign_image)[0]
        sign_img_width = np.shape(sign_image)[1]
        src_radius = random.choice([30, 40, 50, 60])
        transform_image = A.Compose([
            A.RandomSunFlare(p=1.0, flare_roi=(0.3, 0.3, 0.7, 0.7), angle_lower=0, angle_upper=1,
                             num_flare_circles_lower=6,
                             num_flare_circles_upper=10, src_radius=src_radius, src_color=(255, 255, 255)),
            A.Resize(p=1.0, height=sign_img_height, width=sign_img_width, interpolation=1)
        ])
        transformed = transform_image(image=resized_image)
        metadata['type'] = 'sun_flare'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_ringing_overshoot_effect(self, sign_image, resized_image, metadata):
        sign_img_height = np.shape(sign_image)[0]
        sign_img_width = np.shape(sign_image)[1]
        blur_limit = random.choice([5, 7, 9, 11, 13, 15, 17, 19])
        transform_image = A.Compose([
            A.RingingOvershoot(p=1.0, blur_limit=(blur_limit, blur_limit), cutoff=(0.1, 0.2)),
            A.Resize(p=1.0, height=sign_img_height, width=sign_img_width, interpolation=1)
        ])
        transformed = transform_image(image=resized_image)
        metadata['type'] = 'ringing_overshoot'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_perspective_effect(self, sign_image, resized_image, metadata):
        transform_image = A.Compose([
            A.Perspective(p=1.0, scale=(0.15, 0.2), keep_size=True, pad_mode=1, mask_pad_val=0,
                          fit_output=False,
                          interpolation=1)
        ])
        transformed = transform_image(image=sign_image)
        metadata['type'] = 'perspective'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_motion_blur_effect(self, sign_image, resized_image, metadata):
        sign_img_height = np.shape(sign_image)[0]
        sign_img_width = np.shape(sign_image)[1]
        blur_limit = random.choice([15, 21, 25, 31])
        transform_image = A.Compose([
            A.MotionBlur(p=1, blur_limit=(blur_limit, blur_limit), allow_shifted=False),
            A.Resize(p=1.0, height=sign_img_height, width=sign_img_width, interpolation=1)
        ])
        transformed = transform_image(image=resized_image)
        metadata['type'] = 'motion_blur'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_fog_effect(self, sign_image, resized_image, metadata):
        sign_img_height = np.shape(sign_image)[0]
        sign_img_width = np.shape(sign_image)[1]
        gray_value = random.choice([240, 200, 150, 110])
        gray_image = np.full((np.shape(resized_image)[0], np.shape(resized_image)[1], 3), gray_value,
                             dtype=np.uint8)
        transform_image = A.Compose([
            A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.6, alpha_coef=0.2, p=1),
            A.GaussianBlur(blur_limit=(3, 15), p=1),
            A.TemplateTransform(p=1, templates=gray_image),
            A.Resize(p=1.0, height=sign_img_height, width=sign_img_width, interpolation=1)
        ])
        transformed = transform_image(image=resized_image)
        metadata['type'] = 'fog'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_ISO_noise_effect(self, sign_image, resized_image, metadata):
        noise_value = random.choice([0.5, 1, 1.5, 2])
        transform_image = A.Compose([
            A.ISONoise(intensity=(noise_value, noise_value), p=1)
        ])
        transformed = transform_image(image=sign_image)
        metadata['type'] = 'ISO_noise'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_gamma_effect(self, sign_image, resized_image, metadata):
        gamma_value = random.choice([10, 25, 40, 60, 160, 250, 350, 510])
        transform_image = A.Compose([
            A.RandomGamma(gamma_limit=(gamma_value, gamma_value), p=1)
        ])
        transformed = transform_image(image=sign_image)
        metadata['type'] = 'gamma'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def apply_shadow_effect(self, sign_image, resized_image, metadata):
        transform_image = A.Compose([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), num_shadows_lower=None,
                           num_shadows_upper=None, shadow_dimension=5, always_apply=False, p=1)
        ])
        transformed = transform_image(image=sign_image)
        metadata['type'] = 'shadow'
        if self.delete_source:
            self._save_augmented_image_overwriting(transformed['image'], metadata)
        else:
            self._save_augmented_image(transformed['image'], metadata)

    def _save_augmented_image(self, augmented_image, augmented_metadata):
        """
        Saves an augmented image along with its metadata to the dataset directory.

        Args:
            augmented_image (numpy.ndarray): The augmented image to be saved.
            augmented_metadata (dict): Metadata associated with the augmented image. It should contain the following keys:
                - 'class' (str): The class of the image.
                - 'filename' (str): The filename of the image.
                - 'type' (str): Type of transformation applied.
                - 'iteration' (int): Iteration number of the transformation.

        Returns:
            None

        Side Effects:
            - Saves the augmented image to the dataset directory.

        """
        folder_class_path = os.path.join(self.folder_to_load_path, augmented_metadata['class'])
        file_path = os.path.join(folder_class_path,
                                 augmented_metadata['filename'] + '_' + augmented_metadata['transform']['type'] + '_' +
                                 str(augmented_metadata['transform']['iteration']) + '.ppm')
        save_as_ppm(file_path, augmented_image)
    def _save_augmented_image_overwriting(self, augmented_image, augmented_metadata):
        """
        Saves an augmented image along with its metadata to the dataset directory and delete the source image

        Args:
            augmented_image (numpy.ndarray): The augmented image to be saved.
            augmented_metadata (dict): Metadata associated with the augmented image. It should contain the following keys:
                - 'class' (str): The class of the image.
                - 'filename' (str): The filename of the image.
                - 'type' (str): Type of transformation applied.
                - 'iteration' (int): Iteration number of the transformation.

        Returns:
            None

        Side Effects:
            - Saves the augmented image to the dataset directory and delete the source image

        """
        folder_class_path = os.path.join(self.folder_to_load_path, augmented_metadata['class'])
        file_path = os.path.join(folder_class_path,
                                 augmented_metadata['filename'] + '_' + augmented_metadata['type'] + '.ppm')
        save_as_ppm(file_path, augmented_image)
        os.remove(os.path.join(folder_class_path,augmented_metadata['filename']+'.ppm'))

