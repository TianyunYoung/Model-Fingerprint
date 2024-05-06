
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

def slide_crop_image(image, crop_size=(128, 128), stride1=20, stride2=20):
    img_width, img_height = image.size
    crop_width, crop_height = crop_size

    # Initialize the list to hold crops
    crops = []

    # Slide window across the image and crop
    for y in range(0, img_height, stride2):
        for x in range(0, img_width, stride1):
            # Calculate bottom right corner of the crop box
            bottom_right_x = x + crop_width
            bottom_right_y = y + crop_height

            # Check if the crop box is within the image boundaries
            if bottom_right_x <= img_width and bottom_right_y <= img_height:
                crop_box = (x, y, bottom_right_x, bottom_right_y)
                crop = image.crop(crop_box)
                crops.append(crop)

    return crops

class ImageDataset(Dataset):
    def __init__(self, annotations, config, window_slide=False):
        self.data = annotations
        self.config = config
        self.window_slide = window_slide
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, lab = self.data[index]
        img = self.load_sample(img_path)
        lab = torch.tensor(lab, dtype=torch.long)

        return img, lab, img_path

    def load_sample(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        if self.window_slide:
            if img_width > self.config.crop_size[0] and img_height > self.config.crop_size[1]:
                img_crops = slide_crop_image(img, crop_size=self.config.crop_size, \
                                             stride1=(img_width-self.config.crop_size[0])//2, stride2=(img_height-self.config.crop_size[1])//2)
                img_crops_normalized = []
                for img_crop in img_crops:
                    img_crop_normalized = self.norm_transform(img_crop)
                    img_crops_normalized.append(img_crop_normalized.unsqueeze(0))
                return torch.cat(img_crops_normalized)
            else:
                img = transforms.CenterCrop(size=self.config.crop_size)(img)
                img = self.norm_transform(img)
                return torch.cat([img.unsqueeze(0) for i in range(9)])
        else:
            img = transforms.CenterCrop(size=self.config.crop_size)(img)
            img = self.norm_transform(img)
            return img
