import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_folder_dataset(root, img_size=(128, 128)):

    image_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
            
    dataset = datasets.ImageFolder(root=root, transform=image_transform)

    return dataset
