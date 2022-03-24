# from glob import glob
# from random import shuffle
# import pickle
# from tqdm import tqdm


# data_folder_path = "../dataset/retina"
# ratio_train = 0.8  # 一个video分到train的比例，剩下的分到val
# train_select = 0.3  # 对于train数据，每个video文件夹中抽多少来训练
# val_select = 0.2  # 对于val数据，每个video文件夹中抽多少来测试


# dataset_folders = glob(data_folder_path + "/*")
# dataset_folders = [d for d in dataset_folders]  # if ('neural' in d or 'deepfakes' in d or 'original' in d)]

# save_pickle_path = "../dataset/retina_all.pkl"

# data_train = []
# label_train = []
# data_val = []
# label_val = []
# for dataset_folder in dataset_folders:
#     print("-"*15)
#     print(f"Processing {dataset_folder} ...")
#     video_list = glob(dataset_folder + '/*')
#     shuffle(video_list)
#     train_num = int(len(video_list)*ratio_train)
#     video_train = video_list[:train_num]
#     video_val = video_list[train_num:]

#     this_label = 1 if 'original' in dataset_folder else 0  # 1: real, 0: fake

#     print("Getting train data ...")
#     for v_train in tqdm(video_train):
#         img_list = glob(v_train + "/*.png")
#         shuffle(img_list)
#         train_ratio = train_select if 'original' in v_train else train_select*0.26
#         select_num = int(len(img_list)*train_ratio)
#         data_train.extend(img_list[:select_num])
#         label_train.extend([this_label]*select_num)

#     print("Getting val data ...")
#     for v_val in tqdm(video_val):
#         img_list = glob(v_train + "/*.png")
#         shuffle(img_list)
#         val_ratio = val_select if 'original' in v_val else val_select*0.3
#         select_num = int(len(img_list)*val_ratio)
#         data_val.extend(img_list[:select_num])
#         label_val.extend([this_label]*select_num)

# print(f"data_train: {len(data_train)}, data_val: {len(data_val)}")
# pickle.dump((data_train, label_train, data_val, label_val), open(save_pickle_path, "wb"))




from glob import glob
from random import shuffle
import pickle
from tqdm import tqdm


data_folder_path = "../dataset/retina"
save_pickle_path = "../dataset/retina_all.pkl"
ratio_train = 0.8  # 一个video分到train的比例，剩下的分到val
train_select = 0.3  # 对于train数据，每个video文件夹中抽多少来训练
val_select = 0.1  # 对于val数据，每个video文件夹中抽多少来测试
max_img = 500  # 一个视频里最多抽多少图片


dataset_folders = glob(data_folder_path + "/*")

data_train = []
label_train = []
data_val = []
label_val = []


for folder in dataset_folders:
    # fake和real的数据获取比例
    pick_ratio = 0.26 if not 'original' in folder else 0.9
    this_label = 1 if 'original' in folder else 0

    video_folder = glob(folder+"/*")
    shuffle(video_folder)
    train_len = int(len(video_folder)*ratio_train)
    # val_len = int(len(folder)*(1-ratio_train))

    for img_folder in video_folder[:train_len]:
        imgs = glob(img_folder+"/*.png")
        shuffle(imgs)
        # print(f"{img_folder}: {len(imgs)}")
        img_num = min(int(len(imgs)*train_select*pick_ratio), max_img)
        data_train.extend(imgs[:img_num])
        label_train.extend([this_label]*img_num)
    
    for img_folder in video_folder[train_len:]:
        imgs = glob(img_folder+"/*.png")
        shuffle(imgs)
        # print(f"{img_folder}: {len(imgs)}")
        img_num = min(int(len(imgs)*val_select*pick_ratio), max_img)
        data_val.extend(imgs[:img_num])
        label_val.extend([this_label]*img_num)



print(f"data_train: {len(data_train)}")
print(f"    real num: {sum(label_train)}")
print(f"    fake num: {len(data_train)-sum(label_train)}")
print(f"data_val: {len(data_val)}")
print(f"    real num: {sum(label_val)}")
print(f"    fake num: {len(data_val)-sum(label_val)}")
pickle.dump((data_train, label_train, data_val, label_val), open(save_pickle_path, "wb"))
