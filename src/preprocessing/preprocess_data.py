import os
import random
import shutil
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

def reduce_data(src_dir:Path, images_per_class:int):
    '''This function randomly picks reduced images from src_dir and copies them to another folder
    FROM 
        data
         - Train
           -1
           -2
           -3
           -4
    TO
        sample_data
         - Train
           -1
           -2
           -3
           -4

    '''

    # random.seed(42)

    dst_dir = Path("reduced_data") / str(src_dir).split('\\')[-1]

    for class_folder in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_folder)
        if not os.path.isdir(class_path):
            continue  # Skip non-directory files
        
        # List all image files in the folder
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Randomly select images
        
        selected_images = random.sample(images, min(images_per_class, len(images)))
        
        # Create corresponding subfolder in destination
        dst_class_path = os.path.join(dst_dir, class_folder)
        os.makedirs(dst_class_path, exist_ok=True)
        
        # Copy selected images
        for img in selected_images:
            src_img_path = os.path.join(class_path, img)
            dst_img_path = os.path.join(dst_class_path, img)
            shutil.copy2(src_img_path, dst_img_path)
        
        print(f"Copied {len(selected_images)} images from '{class_folder}'")
    

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
        dir_path (str or pathlib.Path): target directory
    
    Returns:
        A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def augment_data():
    '''This function augment the images in training folder of src_dir by using random transformations.
    THe augmentation will only be applied to the training dataset
    FROM 
        data
         - Train
           -1
           -2
           -3
           -4
    TO
        augmented_data
         - Train_300x300
           -1
           -2
           -3
           -4
        - Test_300x300
           -1
           -2
           -3
           -4
    '''

    train_dir_name = "Train_300x300"
    test_dir_name = "Test_300x300"

    train_src_dir = Path("data") / train_dir_name
    test_src_dir = Path("data") / test_dir_name

    dst_dir = Path("data_augmented")
    train_dst_dir = dst_dir / train_dir_name
    test_dst_dir = dst_dir / test_dir_name

    transformations = {'1': [transforms.Compose([transforms.Resize((300,300))]),
                            transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),
                            transforms.Compose([transforms.RandomRotation(15)]),
                            transforms.Compose([transforms.ColorJitter(brightness=(1.3,1.6), contrast=0.4, saturation=0.3)])],
                        '2': [transforms.Compose([transforms.Resize((300,300))]),
                              transforms.Compose([transforms.RandomHorizontalFlip(p=1)])],
                        '3':[transforms.Compose([transforms.Resize((300,300))])],
                        '4':[transforms.Compose([transforms.Resize((300,300))]),
                             transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),
                            transforms.Compose([transforms.RandomRotation(15)]),
                            transforms.Compose([transforms.ColorJitter(brightness=(1.3,1.6), contrast=0.4, saturation=0.3)]),
                            transforms.Compose([transforms.RandomPerspective(0.5, p=1)])] 
                        }
    transformations_name = {'1':['Original', 'HorizontalFlip', 'Rotation', 'ColorJitter'],
                            '2': ['Original', 'HorizontalFlip'],
                            '3': ['Original'],
                            '4': ['Original', 'HorizontalFlip', 'Rotation','ColorJitter','Perspective']}

    train_dst_dir.mkdir(exist_ok=True, parents=True)

    classes = os.listdir(train_src_dir)
    for cl in classes:
        image_folder = train_dst_dir / cl
        image_folder.mkdir(exist_ok=True)

    for cl in classes:
        image_files = os.listdir(train_src_dir / cl)
        print(f"found {len(image_files)} in class {cl} in {train_src_dir / cl}")
        for trans, trans_name in zip(transformations[cl], transformations_name[cl]):
            for image_file in image_files:
                img = Image.open(train_src_dir/cl/image_file)
                transformed_image = trans(img)
                transformed_image.save(f"{train_dst_dir}/{cl}/{image_file[:-5]}_{trans_name}.jpeg")
        
        image_files_after = os.listdir(train_dst_dir / cl)
        print(f"{len(image_files_after)} in class {cl} after data augmentation in {train_dst_dir / cl}")

    
    test_dst_dir.mkdir(exist_ok=True, parents=True)
    classes = os.listdir(test_src_dir)

    for cl in classes:
        image_folder = test_dst_dir / cl
        image_folder.mkdir(exist_ok=True)
    
    for cl in classes:
        image_files = os.listdir(test_src_dir/cl)
        for image_file in image_files:
            img = Image.open(test_src_dir/cl/image_file)
            img.save(f"{test_dst_dir}/{cl}/{image_file}")
        print(f"Moved {len(image_files)} images of class {cl} to augmentation directory")

            
def augment_data_2():
    '''This function augment the images in training folder of src_dir by using random transformations.
    THe augmentation will only be applied to the training dataset
    FROM 
        data
         - Train
           -1
           -2
           -3
           -4
    TO
        augmented_data
         - Train_300x300
           -1
           -2
           -3
           -4
        - Test_300x300
           -1
           -2
           -3
           -4
    '''

    train_dir_name = "Train_300x300"
    test_dir_name = "Test_300x300"

    train_src_dir = Path("data_augmented") / train_dir_name
    test_src_dir = Path("data_augmented") / test_dir_name

    dst_dir = Path("data_augmented_2")
    train_dst_dir = dst_dir / train_dir_name
    test_dst_dir = dst_dir / test_dir_name

    transformations = {'1': [transforms.Compose([transforms.Resize((300,300))]),
                            transforms.Compose([transforms.RandomVerticalFlip(p=1)]),
                            transforms.Compose([transforms.RandomAutocontrast(p=1)]),
                            transforms.RandomResizedCrop(size=300, scale=(0.5,0.6))],
                        '2': [transforms.Compose([transforms.Resize((300,300))]),
                            transforms.Compose([transforms.RandomVerticalFlip(p=1)]),
                            transforms.Compose([transforms.RandomAutocontrast(p=1)]),
                            transforms.RandomResizedCrop(size=300, scale=(0.5,0.6))],
                        '3':[transforms.Compose([transforms.Resize((300,300))]),
                            transforms.Compose([transforms.RandomVerticalFlip(p=1)]),
                            transforms.Compose([transforms.RandomAutocontrast(p=1)]),
                            transforms.RandomResizedCrop(size=300, scale=(0.5,0.6))],
                        '4':[transforms.Compose([transforms.Resize((300,300))]),
                            transforms.Compose([transforms.RandomVerticalFlip(p=1)]),
                            transforms.Compose([transforms.RandomAutocontrast(p=1)]),
                            transforms.RandomResizedCrop(size=300, scale=(0.5,0.6))]
                        }
    transformations_name = {'1':['Original', 'VerticalFlip', 'Autocontrast','ResizedCrop'],
                            '2': ['Original', 'VerticalFlip', 'Autocontrast', 'ResizedCrop'],
                            '3': ['Original', 'VerticalFlip', 'Autocontrast', 'ResizedCrop'],
                            '4': ['Original', 'VerticalFlip', 'Autocontrast', 'ResizedCrop']}

    train_dst_dir.mkdir(exist_ok=True, parents=True)

    classes = os.listdir(train_src_dir)
    for cl in classes:
        image_folder = train_dst_dir / cl
        image_folder.mkdir(exist_ok=True)

    for cl in classes:
        image_files = os.listdir(train_src_dir / cl)
        print(f"found {len(image_files)} in class {cl} in {train_src_dir / cl}")
        for trans, trans_name in zip(transformations[cl], transformations_name[cl]):
            for image_file in image_files:
                img = Image.open(train_src_dir/cl/image_file)
                transformed_image = trans(img)
                transformed_image.save(f"{train_dst_dir}/{cl}/{image_file[:-5]}_{trans_name}.jpeg")
        
        image_files_after = os.listdir(train_dst_dir / cl)
        print(f"{len(image_files_after)} in class {cl} after data augmentation in {train_dst_dir / cl}")

    
    test_dst_dir.mkdir(exist_ok=True, parents=True)
    classes = os.listdir(test_src_dir)

    for cl in classes:
        image_folder = test_dst_dir / cl
        image_folder.mkdir(exist_ok=True)
    
    for cl in classes:
        image_files = os.listdir(test_src_dir/cl)
        for image_file in image_files:
            img = Image.open(test_src_dir/cl/image_file)
            img.save(f"{test_dst_dir}/{cl}/{image_file}")
        print(f"Moved {len(image_files)} images of class {cl} to augmentation directory")



