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
                                'folder/$class_num$'.
        """
        self.dataset_images = []
        self.dataset_path = dataset_path
        self.classes = []

    def load_images(self, classes_to_load=None):
        """
        Loads '.ppm' images from a specified class subdirectory within the dataset path.
        It appends each image along with its metadata (class and filename) to the dataset_images list.
        This method navigates through the structured folder as: 'folder/$class_num$'.
        """
        self.dataset_images = []
        self.classes = classes_to_load if classes_to_load is not None else os.listdir(self.dataset_path)
        for class_folder in tqdm(classes_to_load, desc='Loading classes'):
            class_folder_path = os.path.join(self.dataset_path, class_folder)
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
                            self.dataset_images.append(
                                [img_rgb,
                                 {'class': class_folder,
                                  'filename': os.path.splitext(filename)[0]}
                                 ])
                        else:
                            print(f"Failed to load image: {file_path}")
                self.dataset_images.sort(key=lambda x: (x[1]['class'], x[1]['filename']))

    # def augment_images(self, num_of_total_images=0):
    #     """
    #     Augments images in the dataset to reach a specified total number of images per class.
    #
    #     Args:
    #         num_of_total_images (int): The desired total number of images per class after augmentation.
    #
    #     Returns:
    #         None
    #
    #     Side Effects:
    #         - Augments images in the dataset to meet the desired total number per class.
    #     """
    #     for sign_image, metadata in tqdm(self.dataset_images, desc='Augmenting images'):
    #         metadata = dict(metadata)
    #         # Upscale image for future use
    #         transform_image = A.Compose([
    #             A.SmallestMaxSize(p=1.0, max_size=128, interpolation=1)
    #         ])
    #         resized_image = transform_image(image=sign_image)['image']
    #         sign_img_height = np.shape(sign_image)[0]
    #         sign_img_width = np.shape(sign_image)[1]
    #
    #         # Apply rain effect
    #         # List of parameters
    #         brightness_coefficients = [0.7, 0.5, 0.4, 0.3]
    #         drop_width_values = [1, 2]
    #         blur_values = [5, 7, 10]
    #         rain_types = ['drizzle', 'heavy', 'torrential']
    #         for i1, brightness_coefficient in enumerate(brightness_coefficients):
    #             for i2, drop_width in enumerate(drop_width_values):
    #                 for i3, blur_value in enumerate(blur_values):
    #                     for i4, rain_type in enumerate(rain_types):
    #                         transform_image = A.Compose([
    #                             A.RandomRain(brightness_coefficient=brightness_coefficient,
    #                                          drop_width=drop_width, blur_value=blur_value,
    #                                          p=1, rain_type=rain_type),
    #                             A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
    #                         ])
    #                         transformed = transform_image(image=resized_image)
    #                         metadata['transform'] = {
    #                             'type': 'rain',
    #                             'iteration':
    #                                 ((len(drop_width_values) * len(rain_types) * len(blur_values) * i1) +
    #                                  (len(rain_types) * len(blur_values) * i2) +
    #                                  (len(rain_types) * i3) +
    #                                  i4)}
    #                         self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Spatter effect
    #         spatter_modes = ['rain', 'rain', 'mud', 'mud', 'mud', 'mud']
    #         for index, spatter_mode in enumerate(spatter_modes):
    #             transform_image = A.Compose([
    #                 A.Spatter(p=1.0, mode=spatter_mode)
    #             ])
    #             transformed = transform_image(image=sign_image)
    #             metadata['transform'] = {'type': 'spatter',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Zoom Blur
    #         transform_image = A.Compose([
    #             A.ZoomBlur(p=1.0, max_factor=(1.4, 2), step_factor=(0.03, 0.03))
    #         ])
    #         transformed = transform_image(image=sign_image)
    #         metadata['transform'] = {'type': 'zoom_blur',
    #                                  'iteration': 0}
    #         self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Random sun flare
    #         srcs_radius = [30, 30, 40, 40, 50, 50, 60, 60]
    #         for index, src_radius in enumerate(srcs_radius):
    #             transform_image = A.Compose([
    #                 A.RandomSunFlare(p=1.0, flare_roi=(0.3, 0.3, 0.7, 0.7), angle_lower=0, angle_upper=1,
    #                                  num_flare_circles_lower=6,
    #                                  num_flare_circles_upper=10, src_radius=src_radius, src_color=(255, 255, 255)),
    #                 A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
    #             ])
    #             transformed = transform_image(image=resized_image)
    #             metadata['transform'] = {'type': 'sun_flare',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Ringing overshoot
    #         blur_limits = [5, 7, 9, 11, 13, 15, 17, 19]
    #         for index, blur_limit in enumerate(blur_limits):
    #             transform_image = A.Compose([
    #                 A.RingingOvershoot(p=1.0, blur_limit=(blur_limit, blur_limit), cutoff=(0.1, 0.2)),
    #                 A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
    #             ])
    #             transformed = transform_image(image=resized_image)
    #             metadata['transform'] = {'type': 'ringing_overshoot',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Perspective
    #         for index in range(4):
    #             transform_image = A.Compose([
    #                 A.Perspective(p=1.0, scale=(0.15, 0.2), keep_size=True, pad_mode=1, mask_pad_val=0,
    #                               fit_output=False,
    #                               interpolation=1)
    #             ])
    #             transformed = transform_image(image=sign_image)
    #             metadata['transform'] = {'type': 'perspective',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Motion blur
    #         blur_limits = [15, 21, 25, 31, 15, 21, 25, 31]
    #         for index, blur_limit in enumerate(blur_limits):
    #             transform_image = A.Compose([
    #                 A.MotionBlur(p=1, blur_limit=(blur_limit, blur_limit), allow_shifted=False),
    #                 A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
    #             ])
    #             transformed = transform_image(image=resized_image)
    #             metadata['transform'] = {'type': 'motion_blur',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Fog
    #         gray_values = [240, 200, 150, 110]
    #         for index, gray_value in enumerate(gray_values):
    #             gray_image = np.full((np.shape(resized_image)[0], np.shape(resized_image)[1], 3), gray_value,
    #                                  dtype=np.uint8)
    #             transform_image = A.Compose([
    #                 A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.6, alpha_coef=0.2, p=1),
    #                 A.GaussianBlur(blur_limits=(2, 15), p=1),
    #                 A.TemplateTransform(p=1, templates=gray_image),
    #                 A.ISONoise(intensity=(0.2, 0.5), p=1),
    #                 A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
    #             ])
    #             transformed = transform_image(image=resized_image)
    #             metadata['transform'] = {'type': 'fog',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Noise
    #         noise_values = [0.5, 1, 1.5, 2]
    #         for index, noise_value in enumerate(noise_values):
    #             transform_image = A.Compose([
    #                 A.ISONoise(intensity=(noise_value, noise_value), p=1)
    #             ])
    #             transformed = transform_image(image=sign_image)
    #             metadata['transform'] = {'type': 'ISO_noise',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Gamma values
    #         gamma_values = [10, 25, 40, 60, 160, 250, 350, 510]
    #         for index, gamma_value in enumerate(gamma_values):
    #             transform_image = A.Compose([
    #                 A.RandomGamma(gamma_limit=(gamma_value, gamma_value), p=1)
    #             ])
    #             transformed = transform_image(image=sign_image)
    #             metadata['transform'] = {'type': 'gamma',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))
    #
    #         # Shadow
    #         for index in range(4):
    #             transform_image = A.Compose([
    #                 A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), num_shadows_lower=None,
    #                                num_shadows_upper=None, shadow_dimension=5, always_apply=False, p=1)
    #             ])
    #             transformed = transform_image(image=sign_image)
    #             metadata['transform'] = {'type': 'shadow',
    #                                      'iteration': index}
    #             self._save_augmented_image(transformed['image'], dict(metadata))

    def augment_images(self, num_of_total_images=0):
        """
        Augments images in the dataset to reach a specified total number of images per class.

        Args:
            num_of_total_images (int): The desired total number of images per class after augmentation.

        Returns:
            None

        Side Effects:
            - Augments images in the dataset to meet the desired total number per class.
        """
        # Iterate over each class in the dataset
        for class_label in tqdm(self.classes, desc='Augmenting images'):
            # Filter images belonging to the current class
            class_images = [(img, metadata) for img, metadata in self.dataset_images if metadata['class'] == class_label]

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
                selected_image_and_metadata = random.choice(class_images)

                # Apply the selected augmentation option
                augmented_image, augmented_metadata = self.apply_augmentation(selected_image_and_metadata, selected_option)

                # Save the augmented image and update metadata
                metadata = dict(class_images[0])
                metadata['transform'] = {'type': selected_option, 'iteration': num_existing_images + 1}
                self._save_augmented_image(augmented_image, metadata)

                # Decrement the number of images left to augment
                num_images_to_augment -= 1

    def apply_augmentation(self, selected_image_and_metadata, selected_option):
        sign_image, metadata = selected_image_and_metadata
        metadata = dict(metadata)
        # Upscale image for future use
        transform_image = A.Compose([
            A.SmallestMaxSize(p=1.0, max_size=128, interpolation=1)
        ])
        resized_image = transform_image(image=sign_image)['image']
        sign_img_height = np.shape(sign_image)[0]
        sign_img_width = np.shape(sign_image)[1]



        # Apply rain effect
        # List of parameters
        brightness_coefficients = [0.7, 0.5, 0.4, 0.3]
        drop_width_values = [1, 2]
        blur_values = [5, 7, 10]
        rain_types = ['drizzle', 'heavy', 'torrential']
        for i1, brightness_coefficient in enumerate(brightness_coefficients):
            for i2, drop_width in enumerate(drop_width_values):
                for i3, blur_value in enumerate(blur_values):
                    for i4, rain_type in enumerate(rain_types):
                        transform_image = A.Compose([
                            A.RandomRain(brightness_coefficient=brightness_coefficient,
                                         drop_width=drop_width, blur_value=blur_value,
                                         p=1, rain_type=rain_type),
                            A.Resize(p=1.0, height=sign_img_height, width=sign_img_width, interpolation=1)
                        ])
                        transformed = transform_image(image=resized_image)
                        metadata['transform'] = {
                            'type': 'rain',
                            'iteration':
                                ((len(drop_width_values) * len(rain_types) * len(blur_values) * i1) +
                                 (len(rain_types) * len(blur_values) * i2) +
                                 (len(rain_types) * i3) +
                                 i4)}
                        self._save_augmented_image(transformed['image'], dict(metadata))

        # Spatter effect
        spatter_modes = ['rain', 'rain', 'mud', 'mud', 'mud', 'mud']
        for index, spatter_mode in enumerate(spatter_modes):
            transform_image = A.Compose([
                A.Spatter(p=1.0, mode=spatter_mode)
            ])
            transformed = transform_image(image=sign_image)
            metadata['transform'] = {'type': 'spatter',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))

        # Zoom Blur
        transform_image = A.Compose([
            A.ZoomBlur(p=1.0, max_factor=(1.4, 2), step_factor=(0.03, 0.03))
        ])
        transformed = transform_image(image=sign_image)
        metadata['transform'] = {'type': 'zoom_blur',
                                 'iteration': 0}
        self._save_augmented_image(transformed['image'], dict(metadata))

        # Random sun flare
        srcs_radius = [30, 30, 40, 40, 50, 50, 60, 60]
        for index, src_radius in enumerate(srcs_radius):
            transform_image = A.Compose([
                A.RandomSunFlare(p=1.0, flare_roi=(0.3, 0.3, 0.7, 0.7), angle_lower=0, angle_upper=1,
                                 num_flare_circles_lower=6,
                                 num_flare_circles_upper=10, src_radius=src_radius, src_color=(255, 255, 255)),
                A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
            ])
            transformed = transform_image(image=resized_image)
            metadata['transform'] = {'type': 'sun_flare',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))

        # Ringing overshoot
        blur_limits = [5, 7, 9, 11, 13, 15, 17, 19]
        for index, blur_limit in enumerate(blur_limits):
            transform_image = A.Compose([
                A.RingingOvershoot(p=1.0, blur_limit=(blur_limit, blur_limit), cutoff=(0.1, 0.2)),
                A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
            ])
            transformed = transform_image(image=resized_image)
            metadata['transform'] = {'type': 'ringing_overshoot',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))

        # Perspective
        for index in range(4):
            transform_image = A.Compose([
                A.Perspective(p=1.0, scale=(0.15, 0.2), keep_size=True, pad_mode=1, mask_pad_val=0,
                              fit_output=False,
                              interpolation=1)
            ])
            transformed = transform_image(image=sign_image)
            metadata['transform'] = {'type': 'perspective',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))

        # Motion blur
        blur_limits = [15, 21, 25, 31, 15, 21, 25, 31]
        for index, blur_limit in enumerate(blur_limits):
            transform_image = A.Compose([
                A.MotionBlur(p=1, blur_limit=(blur_limit, blur_limit), allow_shifted=False),
                A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
            ])
            transformed = transform_image(image=resized_image)
            metadata['transform'] = {'type': 'motion_blur',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))

        # Fog
        gray_values = [240, 200, 150, 110]
        for index, gray_value in enumerate(gray_values):
            gray_image = np.full((np.shape(resized_image)[0], np.shape(resized_image)[1], 3), gray_value,
                                 dtype=np.uint8)
            transform_image = A.Compose([
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.6, alpha_coef=0.2, p=1),
                A.GaussianBlur(blur_limits=(2, 15), p=1),
                A.TemplateTransform(p=1, templates=gray_image),
                A.ISONoise(intensity=(0.2, 0.5), p=1),
                A.Resize(p=1.0, height=sign_img_width, width=sign_img_width, interpolation=1)
            ])
            transformed = transform_image(image=resized_image)
            metadata['transform'] = {'type': 'fog',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))

        # Noise
        noise_values = [0.5, 1, 1.5, 2]
        for index, noise_value in enumerate(noise_values):
            transform_image = A.Compose([
                A.ISONoise(intensity=(noise_value, noise_value), p=1)
            ])
            transformed = transform_image(image=sign_image)
            metadata['transform'] = {'type': 'ISO_noise',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))

        # Gamma values
        gamma_values = [10, 25, 40, 60, 160, 250, 350, 510]
        for index, gamma_value in enumerate(gamma_values):
            transform_image = A.Compose([
                A.RandomGamma(gamma_limit=(gamma_value, gamma_value), p=1)
            ])
            transformed = transform_image(image=sign_image)
            metadata['transform'] = {'type': 'gamma',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))

        # Shadow
        for index in range(4):
            transform_image = A.Compose([
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), num_shadows_lower=None,
                               num_shadows_upper=None, shadow_dimension=5, always_apply=False, p=1)
            ])
            transformed = transform_image(image=sign_image)
            metadata['transform'] = {'type': 'shadow',
                                     'iteration': index}
            self._save_augmented_image(transformed['image'], dict(metadata))
        return None, None

    def _save_augmented_image(self, augmented_image, augmented_metadata):
        """
        Saves an augmented image along with its metadata to the dataset directory.

        Args:
            augmented_image (numpy.ndarray): The augmented image to be saved.
            augmented_metadata (dict): Metadata associated with the augmented image. It should contain the following keys:
                - 'class' (str): The class of the image.
                - 'filename' (str): The filename of the image.
                - 'transform' (dict): Transformation information including:
                    - 'type' (str): Type of transformation applied.
                    - 'iteration' (int): Iteration number of the transformation.

        Returns:
            None

        Side Effects:
            - Saves the augmented image to the dataset directory.

        """
        folder_class_path = os.path.join(self.dataset_path, augmented_metadata['class'])
        file_path = os.path.join(folder_class_path,
                                 augmented_metadata['filename'] + '_' + augmented_metadata['transform']['type'] + '_' +
                                 str(augmented_metadata['transform']['iteration']) + '.ppm')
        save_as_ppm(file_path, augmented_image)
