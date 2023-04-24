import os
from PIL import Image


def resize_all_dataset(pics_root_dir,resized_pics_dir,datasets = ['train','test'],classes = ['0','1'],width = 224,height = 224):
    for folder in datasets:
        folder_path = os.path.join(pics_root_dir,folder)#full path of the label folder inside the video folder
        new_image_folder = os.path.join(resized_pics_dir,folder)
        for label in classes:
            label_folder_path = os.path.join(folder_path,label)#full path of the label folder inside the video folder
            full_new_image_folder = os.path.join(new_image_folder,label)
            if os.path.isdir(label_folder_path):
                os.makedirs(full_new_image_folder,exist_ok=True)#making a folder for each label if not exsists already#making a folder for each label if not exsists already
                for image in os.listdir(label_folder_path):
                    image_path = os.path.join(label_folder_path,image)
                    new_image_path = os.path.join(full_new_image_folder,image)
                    if os.path.isfile(image_path) and image!=".DS_Store" and\
                            os.path.isfile(new_image_path) == False:

                        image = Image.open(image_path)
                        new_image = image.resize((width,height))
                        new_image.save(new_image_path)



def resize_folder(pics_root_dir,resized_pics_dir,width = 224,height = 224):
    for image in os.listdir(pics_root_dir):
        image_path = os.path.join(pics_root_dir,image)
        new_image_path = os.path.join(resized_pics_dir,image)
        if os.path.isfile(image_path) and os.path.isfile(new_image_path) == False:
            image = Image.open(image_path)
            new_image = image.resize((width,height))
            new_image.save(new_image_path)
# resize all the pics in the folder
def main():
    base_path = "/home/ido/datasets"
    pics_root_dir = os.path.join(base_path,"blah") #src_folder
    pics_new_root_dir= os.path.join(base_path,"blah2") #dst_folder
    #xml_root_folder = "/home/picky/Documents/datasets/exp_date/30.8.22/labels (copy)" #labels_folder
    width = 256
    height = 256
    one_folder = True
    
    if one_folder :
        resize_folder(pics_root_dir,pics_new_root_dir,
                width = width,height = height)
    else:
        resize_all_dataset(pics_root_dir,pics_new_root_dir,
                datasets = ['train'],classes = ['0'],
                width = width,height = height)


if __name__ == "__main__":
    main()