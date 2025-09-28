import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
import torch.nn.functional as F
import torchvision.models as models
import pickle
import pandas as pd
import os
proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import open_clip
model_path = 'D:/fzh/15-EEG/Neural-MCRL-main/CLIP_checkpoint/open_clip_pytorch_model.bin'  
model_config_path = 'D:/fzh/15-EEG/Neural-MCRL-main/CLIP_checkpoint/open_clip_config.json' 
model_type = 'ViT-H-14'
def resize(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一resize到224×224
        transforms.ToTensor()
    ])
    return transform(img)
def preprocess_mask(img_path, threshold=0.5):
    """对mask进行二值化处理，返回单通道张量"""
    # 打开图像并转为灰度图（单通道）
    with Image.open(img_path).convert("L") as img:
        img = resize(img)
        img_np = np.array(img)
        # 二值化：大于阈值的为1.0，否则为0.0（适配模型输入）
        binary_np = (img_np > threshold).astype(np.float32)
        # 转换为张量并增加通道维度 [H, W] → [1, H, W]
        binary_tensor = torch.from_numpy(binary_np)
        return binary_tensor
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, 
    pretrained=model_path,
    precision='fp32',
    device=device
)

import json



class EEGDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """
    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes = None, pictures = None, val_size=None):

        config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config.json"
        # config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config_meg.json"
        # config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config_fmri.json"
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        data_path = config["data_path"]
        self.img_directory_training = config["img_directory_training"]
        self.img_segment_training = config["img_segment_training"]
        self.img_depth_training = config["img_depth_training"]
        self.img_directory_testing = config["img_directory_testing"]
        self.img_segment_testing = config["img_segment_testing"]
        self.img_depth_testing = config["img_depth_testing"]
        # features_path = config["features_path"]


        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject  
        self.val_size = val_size
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()
        
        self.data = self.extract_eeg(self.data, time_window)
        self.resnet = models.resnet101(pretrained=False)
        # 加载本地权重文件
        weight_path = r"D:\fzh\15-EEG\resnet101-63fe2227.pth"
        self.resnet.load_state_dict(torch.load(weight_path))
        self.resnet_layer4 = nn.Sequential(*list(self.resnet.children())[:8]).to(device)  # 到layer4
        self.resnet_layer4.eval()
        if self.classes is None and self.pictures is None:
        
            text_features_filename = os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_text_features_train.pt') if self.train else os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_text_features_test.pt') # 文本特征文件名
            img_features_filename = os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_img_features_train.pt') if self.train else os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_img_features_test.pt') # 图像特征文件名  

            if os.path.exists(text_features_filename):
                self.text_features = torch.load(text_features_filename)['text_features']
            else:
                self.text_features = self.Textencoder(self.text)  
                torch.save({'text_features': self.text_features.cpu()}, text_features_filename) 

            if os.path.exists(img_features_filename):
                self.img_features = torch.load(img_features_filename)['img_features']
            else:
                self.img_features = self.ImageEncoder(self.img) 
                torch.save({'img_features': self.img_features.cpu()}, img_features_filename)  # 保存图像特征
            
        else:
            self.img_features = self.ImageEncoder(self.img)
            self.text_features = self.Textencoder(self.text)
        
            
    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []
 
        if self.train:
            text_file_path = os.path.join('D:/fzh/15-EEG/Neural-MCRL-main', '4_item_text_train.pkl')
        else:
            text_file_path = os.path.join('D:/fzh/15-EEG/Neural-MCRL-main', '4_item_text_test.pkl')

        if os.path.exists(text_file_path):
            with open(text_file_path, 'rb') as f:
                texts = pickle.load(f) 
        else:
            print(f"Warning: {text_file_path} not found. No text descriptions loaded.")
        
        if self.train:
            directory = self.img_directory_training
        else:
            directory = self.img_directory_testing
        
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        if self.train:
            img_directory = self.img_directory_training
            dep_directory = self.img_depth_training
            seg_directory = self.img_segment_training

        else:
            img_directory = self.img_directory_testing
            dep_directory = self.img_depth_testing
            seg_directory = self.img_segment_testing
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()  

        if self.classes is not None and self.pictures is not None:
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []  
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                seg_path = os.path.join(seg_directory, folder)
                dep_path = os.path.join(dep_directory)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                seg_images = [img for img in os.listdir(seg_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # dep_images = [all_images for img in os.listdir(dep_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                seg_images.sort()
                # dep_images.sort()
                # 确保三类图像数量一致（避免索引越界）

                # 组合路径：每个元素为(原图路径, 分割图路径, 深度图路径)
                for i in range(len(all_images)):
                    img_path = os.path.join(folder_path, all_images[i])
                    seg_img_path = os.path.join(seg_path, seg_images[i])
                    dep_img_path = os.path.join(dep_path, all_images[i])
                    # 将三个路径作为元组添加到images列表
                    images.append( (img_path, seg_img_path, dep_img_path) )
        else:
            
            print("Error")
            
        print("self.subjects", self.subjects)
        print("exclude_subject", self.exclude_subject)
        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:  
                    continue            
                # print("subject:", subject)    
                file_name = 'preprocessed_eeg_training.npy'

                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()                
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']  

                n_classes = 1654  
                samples_per_class = 10  
                
                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):  
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]  
                            labels = torch.full((1,), c, dtype=torch.long).detach()  
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)  

                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()  
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        # if self.exclude_subject==None:
                        #     preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # else:
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 1)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                 
            else:
                if subject == self.exclude_subject or self.exclude_subject==None:  
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']  
                    n_classes = 200  # Each class contains 1 images
                    
                    samples_per_class = 1  

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:  # If we've defined specific classes and the current class is not in the list, skip
                            continue
                        start_index = i * samples_per_class  # Update start_index for each class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  # Add labels to the label list
                else:
                    continue
        # datalist: (subjects * classes) * (10 * 4 * 17 * 100)
        # data_tensor: (subjects * classes * 10 * 4) * 17 * 100
        # data_list = np.mean(data_list, )
        # print("data_list", len(data_list))
        if self.train:
            # print("data_list", *data_list[0].shape[1:])            
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # print("label_tensor", label_tensor.shape)
            print("data_tensor", data_tensor.shape)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # label_tensor = torch.cat(label_list, dim=0)
            # print("label_tensor", label_tensor.shape)
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
        # print("data_tensor", data_tensor.shape)
        # label_list: (subjects * classes) * 10
        # label_tensor: (subjects * classes * 10)
        # print("label_tensor = torch.cat(label_list, dim=0)")
        # print(label_list)
        label_tensor = torch.cat(label_list, dim=0)
        # label_tensor = torch.cat(label_list, dim=0)
        # print(label_tensor[:300])
        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)        
                mapping = {val.item(): index for index, val in enumerate(unique_values)}   
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        else:
            # label_tensor = label_tensor.repeat_interleave(80)
            # if self.classes is not None:
            #     unique_values = torch.unique(label_tensor, sorted=False)
           
            #     mapping = {val.item(): index for index, val in enumerate(torch.flip(unique_values, [0]))}
            #     label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
            pass      

                    
        self.times = times
        self.ch_names = ch_names

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        # print("self.times", self.times.shape)
        # print("indices", indices)
        # print("indices", indices.shape)
        # print("eeg_data", eeg_data.shape)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]
        # print(f"extracted_data shape: {extracted_data.shape}")

        return extracted_data
    

    def Textencoder(self, text):   
        # 输入维度 为列表 N, 4
        batch_size = 32  # number of text *groups*
        text_features_list = []

        # assert len(text) % 4 == 0, "Total number of texts should be a multiple of 4"
        grouped_text = [text[i:i+1][0][0] for i in range(0, len(text))]  # 每组4句话
        total_groups = len(grouped_text)

        for i in range(0, total_groups, batch_size):
            batch_groups = grouped_text[i:i + batch_size]  # 取 batch_size 个 group
            flat_texts = [sentence for group in batch_groups for sentence in group]  # 展平为一维
            text_inputs = torch.cat([open_clip.tokenize(t) for t in flat_texts]).to(device)

            with torch.no_grad():
                text_features = vlmodel.encode_text(text_inputs)

            text_features = F.normalize(text_features, dim=-1).detach()
            text_features = text_features.view(-1, 4, 1024)  # [batch_size, 4, 1024]
            text_features_list.append(text_features.cpu()) 

        all_text_features = torch.cat(text_features_list, dim=0)  # [total_groups, 4, 1024]

        print(f"Text features shape: {all_text_features.shape}")
        return all_text_features

        
        
    # def ImageEncoder(self,images):
    #     batch_size = 20  
    #     image_features_list = []
      
    #     for i in range(0, len(images), batch_size):
    #         batch_images = images[i:i + batch_size]
    #         image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

    #         with torch.no_grad():
    #             batch_image_features = vlmodel.encode_image(image_inputs) # torch.Size([20, 1024]) torch.Size([20, 3, 224, 224])
    #             batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

    #         image_features_list.append(batch_image_features)

    #     image_features = torch.cat(image_features_list, dim=0)
        
    #     return image_features
    def ImageEncoder(self, images):
        """
        images: [B, 3, 500, 500]
        depth: [B, 3, 500, 500]
        mask:  [B, 1, 500, 500]
        """

        
        batch_size = 20
        image_features_list = []
        for i in range(0, len(images), batch_size):

            image1 = images[i:i + batch_size]
            image = [tup[0] for tup in image1]
            depth = [tup[2] for tup in image1]
            mask = [tup[1] for tup in image1]
            
            image = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in image]).to(device)
            depth = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in depth]).to(device)
            mask = torch.stack([preprocess_mask(img) for img in mask]).to(device) 
            # 图像预处理后提取原始视觉embedding
            masked_images = image * mask
            image_inputs = F.interpolate(masked_images, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)  # [B, 1024]
                batch_image_features_po = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)

            inverted_masks = 1 - mask  # 关键修改：通过1减去原mask实现取反
            
            # 2. 应用取反后的mask
            masked_images = image * inverted_masks  # [B, 3, H, W]，原0区域保留，1区域屏蔽
            image_inputs = F.interpolate(masked_images, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)  # [B, 1024]
                batch_image_features_ba = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
            # 纹理特征提取


            with torch.no_grad():
                resnet_feats = self.resnet_layer4(image)  # [B, 2048, H/32, W/32] → [B, 2048, 16, 16]
                global_feats = torch.mean(resnet_feats, dim=[2, 3], keepdim=True)
                texture_feat = global_feats[:, :1024] + global_feats[:, 1024:]  # [B, 1024]  # 无参数操作
                texture_feat = texture_feat.squeeze()
                texture_feat = texture_feat / texture_feat.norm(dim=-1, keepdim=True)
            # 深度图映射
            depth = 0.299 * depth[:, 0:1, :, :] + 0.587 * depth[:, 1:2, :, :] + 0.114 * depth[:, 2:3, :, :]  # [B, 1, H, W]
            depth_down = F.adaptive_avg_pool2d(depth, (32, 32))  # [B, 1, 16, 16]
            depth_flat = depth_down.view(batch_size, -1)  # [B, 3*16*16]
            # depth_feat = F.linear(depth_flat, depth_linear_weight)  # [B, 1024]
            depth_feat = depth_flat / depth_flat.norm(dim=-1, keepdim=True)

            # images图映射
            images_down = F.adaptive_avg_pool2d(image, (32, 32))  # [B, 1, 16, 16]
            images_flat = images_down.view(batch_size, -1, 1024)  # [B, 1*16*16]
            images_flat = images_flat[:, 0, :] + images_flat[:, 1, :] + images_flat[:, 2, :]
            # images_feat = F.linear(images_flat, images_linear_weight)  # [B, 1024]
            images_feat = images_flat / images_flat.norm(dim=-1, keepdim=True)

            # 拼接四个模态
            image_features = torch.stack([batch_image_features_po, batch_image_features_ba, depth_feat, images_feat, texture_feat], dim=1)  # [B, 4, 1024]
            image_features_list.append(image_features)
        image_features = torch.cat(image_features_list, dim=0)
        return image_features
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 10 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
            else:
                text_index = (index % index_n_sub_test)
                text_index = text_index % 200
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
                img_index = img_index % 200
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
        text = self.text[text_index]
        img = self.img[img_index]
        
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        
        return x, label, text, text_features, img, img_features

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same
class MEGDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
    """
    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes = None, pictures = None, val_size=None):








        # config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config.json"
        config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config_meg.json"
        # config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config_fmri.json"
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        data_path = config["data_path"]
        self.img_directory_training = config["img_directory_training"]
        self.img_segment_training = config["img_segment_training"]
        self.img_depth_training = config["img_depth_training"]
        self.img_directory_testing = config["img_directory_testing"]
        self.img_segment_testing = config["img_segment_testing"]
        self.img_depth_testing = config["img_depth_testing"]
        # features_path = config["features_path"]




        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject  
        self.val_size = val_size
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()
        
        # self.data = self.extract_eeg(self.data, time_window)
        self.resnet = models.resnet101(pretrained=False)
        # 加载本地权重文件
        weight_path = r"D:\fzh\15-EEG\resnet101-63fe2227.pth"
        self.resnet.load_state_dict(torch.load(weight_path))
        self.resnet_layer4 = nn.Sequential(*list(self.resnet.children())[:8]).to(device)  # 到layer4
        self.resnet_layer4.eval()
        if self.classes is None and self.pictures is None:
        
            text_features_filename = os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_text_meg_features_train.pt') if self.train else os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_text_meg_features_test.pt') # 文本特征文件名
            img_features_filename = os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_img_meg_features_train.pt') if self.train else os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_img_meg_features_test.pt') # 图像特征文件名  

            if os.path.exists(text_features_filename):
                self.text_features = torch.load(text_features_filename)['text_features']
            else:
                self.text_features = self.Textencoder(self.text)  
                torch.save({'text_features': self.text_features.cpu()}, text_features_filename) 

            if os.path.exists(img_features_filename):
                self.img_features = torch.load(img_features_filename)['img_features']
            else:
                self.img_features = self.ImageEncoder(self.img) 
                torch.save({'img_features': self.img_features.cpu()}, img_features_filename)  # 保存图像特征
            
        else:
            self.img_features = self.ImageEncoder(self.img)
            self.text_features = self.Textencoder(self.text)
        
            
    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []
 
        if self.train:
            text_file_path = 'D:/fzh/15-EEG/meg_4_item_text_train.pkl'
        else:
            text_file_path = 'D:/fzh/15-EEG/meg_4_item_text_test.pkl'

        if os.path.exists(text_file_path):
            with open(text_file_path, 'rb') as f:
                texts = pickle.load(f) 
        else:
            print(f"Warning: {text_file_path} not found. No text descriptions loaded.")
        
        if self.train:
            directory = self.img_directory_training
        else:
            directory = self.img_directory_testing
        
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        if self.train:
            img_directory = self.img_directory_training
            dep_directory = self.img_depth_training
            seg_directory = self.img_segment_training

        else:
            img_directory = self.img_directory_testing
            dep_directory = self.img_depth_testing
            seg_directory = self.img_segment_testing
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()  

        if self.classes is not None and self.pictures is not None:
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []  
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                seg_path = os.path.join(seg_directory, folder)
                dep_path = os.path.join(dep_directory)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                seg_images = [img for img in os.listdir(seg_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # dep_images = [all_images for img in os.listdir(dep_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                seg_images.sort()
                # dep_images.sort()
                # 确保三类图像数量一致（避免索引越界）

                # 组合路径：每个元素为(原图路径, 分割图路径, 深度图路径)
                for i in range(len(all_images)):
                    img_path = os.path.join(folder_path, all_images[i])
                    seg_img_path = os.path.join(seg_path, seg_images[i])
                    dep_img_path = os.path.join(dep_path, all_images[i])
                    # 将三个路径作为元组添加到images列表
                    images.append( (img_path, seg_img_path, dep_img_path) )
        else:
            
            print("Error")
            
        print("self.subjects", self.subjects)
        print("exclude_subject", self.exclude_subject)
        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:  
                    continue            
                # print("subject:", subject)    
                file_name = 'train.pt'

                file_path = os.path.join(self.data_path, subject, file_name)
                data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
                preprocessed_eeg_data = torch.from_numpy(data['eeg']).float().detach()                
                # times = torch.from_numpy(data['times']).detach()[50:]
                # ch_names = data['ch_names']  

                n_classes = 1654  
                samples_per_class = 12
                
                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):  
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]  
                            labels = torch.full((1,), c, dtype=torch.long).detach()  
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)  

                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()  
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        # if self.exclude_subject==None:
                        #     preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # else:
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 1)
                        # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                 
            else:
                if subject == self.exclude_subject or self.exclude_subject==None:  
                    file_name = 'test.pt'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
                    preprocessed_eeg_data = torch.from_numpy(data['eeg']).float().detach()         
                    # times = torch.from_numpy(data['times']).detach()[50:]
                    # ch_names = data['ch_names']  
                    n_classes = 200  # Each class contains 1 images
                    
                    samples_per_class = 1  

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:  # If we've defined specific classes and the current class is not in the list, skip
                            continue
                        start_index = i * samples_per_class  # Update start_index for each class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  # Add labels to the label list
                else:
                    continue
        # datalist: (subjects * classes) * (10 * 4 * 17 * 100)
        # data_tensor: (subjects * classes * 10 * 4) * 17 * 100
        # data_list = np.mean(data_list, )
        # print("data_list", len(data_list))
        if self.train:
            # print("data_list", *data_list[0].shape[1:])            
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # print("label_tensor", label_tensor.shape)
            print("data_tensor", data_tensor.shape)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # label_tensor = torch.cat(label_list, dim=0)
            # print("label_tensor", label_tensor.shape)
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
        # print("data_tensor", data_tensor.shape)
        # label_list: (subjects * classes) * 10
        # label_tensor: (subjects * classes * 10)
        # print("label_tensor = torch.cat(label_list, dim=0)")
        # print(label_list)
        label_tensor = torch.cat(label_list, dim=0)
        # label_tensor = torch.cat(label_list, dim=0)
        # print(label_tensor[:300])
        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            # label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)        
                mapping = {val.item(): index for index, val in enumerate(unique_values)}   
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        else:
            # label_tensor = label_tensor.repeat_interleave(80)
            # if self.classes is not None:
            #     unique_values = torch.unique(label_tensor, sorted=False)
           
            #     mapping = {val.item(): index for index, val in enumerate(torch.flip(unique_values, [0]))}
            #     label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
            pass      

                    
        # self.times = times
        # self.ch_names = ch_namesC

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        # print("self.times", self.times.shape)
        # print("indices", indices)
        # print("indices", indices.shape)
        # print("eeg_data", eeg_data.shape)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]
        # print(f"extracted_data shape: {extracted_data.shape}")

        return extracted_data
    

    def Textencoder(self, text):   
        # 输入维度 为列表 N, 4
        batch_size = 32  # number of text *groups*
        text_features_list = []

        # assert len(text) % 4 == 0, "Total number of texts should be a multiple of 4"
        grouped_text = [text[i:i+1][0][0] for i in range(0, len(text))]  # 每组4句话
        total_groups = len(grouped_text)

        for i in range(0, total_groups, batch_size):
            batch_groups = grouped_text[i:i + batch_size]  # 取 batch_size 个 group
            flat_texts = [sentence for group in batch_groups for sentence in group]  # 展平为一维
            text_inputs = torch.cat([open_clip.tokenize(t) for t in flat_texts]).to(device)

            with torch.no_grad():
                text_features = vlmodel.encode_text(text_inputs)

            text_features = F.normalize(text_features, dim=-1).detach()
            text_features = text_features.view(-1, 4, 1024)  # [batch_size, 4, 1024]
            text_features_list.append(text_features.cpu()) 

        all_text_features = torch.cat(text_features_list, dim=0)  # [total_groups, 4, 1024]

        print(f"Text features shape: {all_text_features.shape}")
        return all_text_features

        
        
    # def ImageEncoder(self,images):
    #     batch_size = 20  
    #     image_features_list = []
      
    #     for i in range(0, len(images), batch_size):
    #         batch_images = images[i:i + batch_size]
    #         image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

    #         with torch.no_grad():
    #             batch_image_features = vlmodel.encode_image(image_inputs) # torch.Size([20, 1024]) torch.Size([20, 3, 224, 224])
    #             batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

    #         image_features_list.append(batch_image_features)

    #     image_features = torch.cat(image_features_list, dim=0)
        
    #     return image_features
    def ImageEncoder(self, images):
        """
        images: [B, 3, 500, 500]
        depth: [B, 3, 500, 500]
        mask:  [B, 1, 500, 500]
        """

        
        batch_size = 8
        image_features_list = []
        for i in range(0, len(images), batch_size):

            image1 = images[i:i + batch_size]
            image = [tup[0] for tup in image1]
            depth = [tup[2] for tup in image1]
            mask = [tup[1] for tup in image1]
            
            image = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in image]).to(device)
            depth = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in depth]).to(device)
            mask = torch.stack([preprocess_mask(img) for img in mask]).to(device) 
            # 图像预处理后提取原始视觉embedding
            masked_images = image * mask
            image_inputs = F.interpolate(masked_images, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)  # [B, 1024]
                batch_image_features_po = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)

            inverted_masks = 1 - mask  # 关键修改：通过1减去原mask实现取反
            
            # 2. 应用取反后的mask
            masked_images = image * inverted_masks  # [B, 3, H, W]，原0区域保留，1区域屏蔽
            image_inputs = F.interpolate(masked_images, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)  # [B, 1024]
                batch_image_features_ba = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
            # 纹理特征提取


            with torch.no_grad():
                resnet_feats = self.resnet_layer4(image)  # [B, 2048, H/32, W/32] → [B, 2048, 16, 16]
                global_feats = torch.mean(resnet_feats, dim=[2, 3], keepdim=True)
                texture_feat = global_feats[:, :1024] + global_feats[:, 1024:]  # [B, 1024]  # 无参数操作
                texture_feat = texture_feat.squeeze()
                texture_feat = texture_feat / texture_feat.norm(dim=-1, keepdim=True)
            # 深度图映射
            depth = 0.299 * depth[:, 0:1, :, :] + 0.587 * depth[:, 1:2, :, :] + 0.114 * depth[:, 2:3, :, :]  # [B, 1, H, W]
            depth_down = F.adaptive_avg_pool2d(depth, (32, 32))  # [B, 1, 16, 16]
            depth_flat = depth_down.view(batch_size, -1)  # [B, 3*16*16]
            # depth_feat = F.linear(depth_flat, depth_linear_weight)  # [B, 1024]
            depth_feat = depth_flat / depth_flat.norm(dim=-1, keepdim=True)

            # images图映射
            images_down = F.adaptive_avg_pool2d(image, (32, 32))  # [B, 1, 16, 16]
            images_flat = images_down.view(batch_size, -1, 1024)  # [B, 1*16*16]
            images_flat = images_flat[:, 0, :] + images_flat[:, 1, :] + images_flat[:, 2, :]
            # images_feat = F.linear(images_flat, images_linear_weight)  # [B, 1024]
            images_feat = images_flat / images_flat.norm(dim=-1, keepdim=True)

            # 拼接四个模态
            image_features = torch.stack([batch_image_features_po, batch_image_features_ba, depth_feat, images_feat, texture_feat], dim=1)  # [B, 4, 1024]
            image_features_list.append(image_features)
        image_features = torch.cat(image_features_list, dim=0)
        return image_features
    def __getitem__(self, index):
        # # Get the data and label corresponding to "index"
        # # index: (subjects * classes * 10 * 4)
        # x = self.data[index]
        # label = self.labels[index]
        
        # if self.pictures is None:
        #     if self.classes is None:
        #         index_n_sub_train = self.n_cls * 10 * 4
        #         index_n_sub_test = self.n_cls * 1 * 80
        #     else:
        #         index_n_sub_test = len(self.classes)* 1 * 80
        #         index_n_sub_train = len(self.classes)* 10 * 4
        #     # text_index: classes
        #     if self.train:
        #         text_index = (index % index_n_sub_train) // (10 * 4)
        #     else:
        #         text_index = (index % index_n_sub_test)
        #     # img_index: classes * 10
        #     if self.train:
        #         img_index = (index % index_n_sub_train) // (4)
        #     else:
        #         img_index = (index % index_n_sub_test)
        # else:
        #     if self.classes is None:
        #         index_n_sub_train = self.n_cls * 1 * 4
        #         index_n_sub_test = self.n_cls * 1 * 80
        #     else:
        #         index_n_sub_test = len(self.classes)* 1 * 80
        #         index_n_sub_train = len(self.classes)* 1 * 4
        #     # text_index: classes
        #     if self.train:
        #         text_index = (index % index_n_sub_train) // (1 * 4)
        #     else:
        #         text_index = (index % index_n_sub_test)
        #     # img_index: classes * 10
        #     if self.train:
        #         img_index = (index % index_n_sub_train) // (4)
        #     else:
        #         img_index = (index % index_n_sub_test)
        # text = self.text[text_index]
        # img = self.img[img_index]
        
        # text_features = self.text_features[text_index]
        # img_features = self.img_features[img_index]
        
        # return x, label, text, text_features, img, img_features
        # """Get item by index"""
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 12 * 1
                index_n_sub_test = self.n_cls * 1 * 12
            else:
                index_n_sub_test = len(self.classes)* 1 * 12
                index_n_sub_train = len(self.classes)* 12 * 1
                
            # Calculate text and image indices
            if self.train:
                text_index = (index % index_n_sub_train) // (12 * 1)
                img_index = (index % index_n_sub_train) // (1)
            else:
                text_index = (index % index_n_sub_test) // (1)
                img_index = (index % index_n_sub_test) // (1)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 1
                index_n_sub_test = self.n_cls * 1 * 12
            else:
                index_n_sub_test = len(self.classes)* 1 * 12
                index_n_sub_train = len(self.classes)* 1 * 1
                
            if self.train:
                text_index = (index % index_n_sub_train) // (1)
                img_index = (index % index_n_sub_train) // (1)
            else:
                text_index = (index % index_n_sub_test) // (1)
                img_index = (index % index_n_sub_test) // (1)
                
        text = self.text[text_index]
        
        # if self.use_caption:
        #     text_features = torch.zeros((1, 1, 1024))
        # else:
        text_features = self.text_features[text_index]        
            
        if self.train:
            img_features = self.img_features[img_index]
            img = self.img[img_index]
        else:
            img_features = self.img_features[img_index]
            img = self.img[img_index]        
            
        return x, label, text, text_features, img, img_features

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same
class fMRIDataset():
    """
    subjects = ['sub-01', 'sub-02', 'sub-03']
    """
    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes = None, pictures = None, val_size=None):







        # config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config.json"
        # config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config_meg.json"
        config_path = "D:/fzh/15-EEG/Neural-MCRL-main/EEGToVisual/data_config_fmri.json"
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        self.data_path = config["data_path"]
        self.img_segment = config["img_segment"]
        self.img_directory = config["img_directory"]
        self.img_depth = config["img_depth"]
        # features_path = config["features_path"]




        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 720 if train else 100
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject  
        self.val_size = val_size
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()
        
        # self.data = self.extract_eeg(self.data, time_window)
        self.resnet = models.resnet101(pretrained=False)
        # 加载本地权重文件
        weight_path = r"D:\fzh\15-EEG\resnet101-63fe2227.pth"
        self.resnet.load_state_dict(torch.load(weight_path))
        self.resnet_layer4 = nn.Sequential(*list(self.resnet.children())[:8]).to(device)  # 到layer4
        self.resnet_layer4.eval()
        if self.classes is None and self.pictures is None:
        
            text_features_filename = os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_text_fmri_features_train.pt') if self.train else os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_text_fmri_features_test.pt') # 文本特征文件名
            img_features_filename = os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_img_fmri_features_train.pt') if self.train else os.path.join(f'D:/fzh/15-EEG/Neural-MCRL-main', f'{model_type}_img_fmri_features_test.pt') # 图像特征文件名  

            if os.path.exists(text_features_filename):
                self.text_features = torch.load(text_features_filename)['text_features']
            else:
                self.text_features = self.Textencoder(self.text)  
                torch.save({'text_features': self.text_features.cpu()}, text_features_filename) 

            if os.path.exists(img_features_filename):
                self.img_features = torch.load(img_features_filename)['img_features']
            else:
                self.img_features = self.ImageEncoder(self.img) 
                torch.save({'img_features': self.img_features.cpu()}, img_features_filename)  # 保存图像特征
            
        else:
            self.img_features = self.ImageEncoder(self.img)
            self.text_features = self.Textencoder(self.text)
        
            
    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []
 
        text_file_path = 'D:/fzh/15-EEG/fMRI_4_item_text_train_test_mix.pkl'

        if os.path.exists(text_file_path):
            with open(text_file_path, 'rb') as f:
                texts = pickle.load(f) 
        else:
            print(f"Warning: {text_file_path} not found. No text descriptions loaded.")


        subject_csv_map = {
            "sub-01": r"D:/fzh/15-EEG/sub-01_StimulusMetadata.csv",
            "sub-02": r"D:/fzh/15-EEG/sub-02_StimulusMetadata.csv",
            "sub-03": r"D:/fzh/15-EEG/sub-03_StimulusMetadata.csv"
        }

        subject_file_order = {}
        subject_file_set = {}
        for subject, csv_path in subject_csv_map.items():
            df = pd.read_csv(csv_path)
            # 取第一列train/test, 第六列图片文件名
            df_train = df[df.iloc[:,0]=='train']  # 选train行
            df_test  = df[df.iloc[:,0]=='test']   # 选test行
            # 保存按图片名排序的索引
            subject_file_order[subject] = {
                'train': df_train.iloc[:,5].argsort().values,
                'test' : df_test.iloc[:,5].argsort().values
            }
        for subject, csv_path in subject_csv_map.items():
            df = pd.read_csv(csv_path)
            # 根据 train/test 筛选
            train_files = set(df[df.iloc[:,0]=='train'].iloc[:,5].values)
            test_files  = set(df[df.iloc[:,0]=='test'].iloc[:,5].values)
            subject_file_set[subject] = {
                'train': train_files,
                'test' : test_files
            }
        # a = [5, 5, 2, 0, 0, 1, 1, 1, 0, 5, 0, 2, 2, 2, 5, 5, 0, 2, 4, 0, 2, 2, 2, 2, 4, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 0, 2, 2, 0, 1, 5, 1, 2, 2, 2, 0, 1, 1, 5, 1, 0, 1, 1, 1, 5, 1, 2, 2, 2, 2, 1, 2, 4, 2, 2, 4, 1, 3, 2, 2, 0, 5, 2, 0, 5, 5, 2, 5, 2, 2, 2, 5, 0, 3, 1, 2, 2, 2, 3, 3, 2, 0, 3, 1, 4, 3, 2, 3, 2, 3, 2, 2, 2, 2, 1, 0, 2, 2, 2, 4, 0, 5, 3, 1, 5, 4, 1, 1, 1, 2, 1, 2, 1, 1, 0, 0, 2, 1, 5, 0, 1, 2, 1, 2, 5, 0, 2, 1, 2, 1, 1, 2, 0, 2, 3, 2, 5, 3, 4, 2, 3, 1, 2, 5, 0, 2, 1, 5, 2, 2, 5, 1, 2, 0, 2, 2, 2, 2, 4, 1, 2, 2, 3, 2, 3, 5, 0, 0, 0, 5, 5, 1, 1, 5, 3, 2, 2, 0, 2, 2, 2, 2, 3, 2, 2, 5, 2, 4, 1, 0, 1, 0, 3, 2, 2, 0, 2, 2, 5, 2, 3, 2, 4, 0, 1, 0, 2, 1, 2, 2, 2, 2, 4, 2, 2, 3, 2, 2, 1, 5, 2, 0, 0, 5, 2, 4, 2, 2, 4, 0, 1, 2, 2, 1, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 0, 1, 5, 5, 1, 2, 2, 3, 5, 5, 2, 2, 2, 0, 1, 2, 1, 0, 2, 4, 3, 0, 2, 2, 1, 2, 5, 5, 2, 5, 2, 2, 2, 5, 2, 5, 0, 3, 2, 3, 1, 3, 5, 1, 5, 4, 1, 0, 2, 5, 1, 2, 1, 3, 2, 2, 2, 4, 5, 2, 0, 0, 2, 4, 2, 1, 5, 0, 2, 2, 2, 1, 2, 2, 3, 5, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 3, 3, 1, 1, 2, 5, 1, 2, 0, 1, 2, 0, 2, 1, 0, 2, 1, 0, 5, 5, 2, 2, 1, 5, 0, 5, 3, 1, 4, 4, 1, 2, 0, 5, 2, 3, 2, 3, 2, 2, 5, 5, 0, 0, 1, 0, 2, 1, 2, 2, 2, 3, 1, 2, 2, 0, 1, 2, 2, 1, 2, 1, 0, 4, 1, 2, 0, 1, 5, 2, 5, 1, 3, 0, 1, 2, 0, 1, 2, 2, 2, 1, 0, 1, 5, 4, 2, 0, 1, 1, 5, 5, 2, 2, 0, 2, 2, 1, 2, 1, 2, 2, 3, 5, 3, 5, 2, 3, 1, 1, 2, 1, 2, 2, 5, 2, 5, 3, 0, 4, 5, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 5, 1, 2, 3, 4, 0, 5, 2, 2, 3, 0, 0, 1, 2, 4, 0, 2, 3, 5, 2, 0, 2, 2, 0, 1, 2, 1, 2, 5, 5, 5, 0, 5, 2, 5, 4, 4, 2, 1, 5, 2, 1, 3, 5, 1, 1, 2, 3, 2, 2, 1, 2, 3, 5, 2, 1, 0, 0, 2, 2, 2, 2, 2, 0, 1, 0, 2, 2, 4, 3, 1, 3, 2, 5, 1, 2, 1, 2, 2, 5, 2, 0, 3, 0, 2, 2, 3, 1, 3, 2, 0, 4, 5, 2, 2, 1, 0, 2, 1, 5, 5, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0, 2, 0, 4, 2, 2, 1, 0, 1, 3, 2, 2, 5, 1, 2, 2, 1, 2, 2, 0, 3, 1, 2, 0, 1, 5, 1, 1, 1, 0, 2, 2, 5, 1, 1, 4, 1, 5, 2, 3, 2, 2, 2, 0, 2, 2, 2, 0, 2, 1, 1, 1, 1, 4, 5, 0, 5, 2, 5, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 3, 2, 3, 1, 0, 3, 0, 3, 2, 2, 4, 5, 2, 0, 4, 5, 1, 2, 2, 2, 0, 2, 2, 2, 1, 1, 5, 4, 2, 2, 0, 2, 5, 5, 1, 5, 2, 3, 0, 2, 5, 4, 0, 0, 0, 0, 2, 3, 0, 5, 2, 2, 2, 3, 3, 0, 1, 2, 2, 1, 1, 2, 2, 4, 2, 2, 2, 2, 2, 2, 0, 3, 2, 2, 1, 2, 5, 2, 0, 2, 5, 3, 1, 5, 1, 1, 2, 0, 5, 5, 1, 5, 1, 0, 2, 0, 2, 4, 5, 4, 1, 1, 2, 2, 1, 1, 1, 5, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 3, 2, 5, 5, 1, 3, 2, 2, 2, 3, 1, 2, 0, 0, 2, 5, 1, 4, 2, 1, 0, 0, 4, 2, 1, 5, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 0, 3, 2, 1, 1, 0, 2, 0, 5, 2, 2, 1, 1, 1, 5, 2, 2, 5, 2, 1, 2, 2, 0, 1, 0, 5, 2, 0, 5, 5, 2, 1, 1, 5, 0, 2, 1, 2, 3, 1, 2, 2, 1, 2, 3, 2, 2, 0, 1, 3, 2, 5, 5, 1, 2, 1, 2, 2, 2, 3, 3, 4, 2, 5, 5, 2, 1, 0, 1, 4, 1, 2, 3, 5, 0, 3, 2, 2, 2, 0, 2, 0, 2, 2, 4, 2, 1, 5, 0, 2, 3, 2, 5, 2, 0, 5, 1, 2, 1, 1, 2, 0, 4, 2, 1, 3, 1, 0, 2, 2, 3, 2, 0, 2, 0, 1, 0, 5, 5, 1, 2, 2, 1, 0, 2, 2, 5, 1, 1, 0, 2, 4, 5, 5, 3, 1, 0, 2, 5, 2, 5, 3, 0, 1, 0, 2, 3, 1, 2, 1, 2, 0, 5, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 5, 1, 5, 2, 2, 4, 1, 2, 2, 2, 2, 4, 1, 3, 3, 2, 2, 2, 1, 1, 2, 5, 2, 1, 2, 3, 2, 1, 2, 1, 4, 2, 2, 1, 0, 1, 2, 2, 0, 2, 5, 5, 3, 0, 3, 0, 2, 2, 2, 2, 5, 0, 3, 1, 5, 1, 2, 5, 4, 0, 2, 1, 3, 1, 1, 3, 1, 4, 5, 2, 2, 2, 1, 0, 2, 1, 5, 2, 4, 1, 1, 5, 2, 0, 2, 0, 3, 2, 2, 0, 3, 4, 5, 5, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 1, 1, 1, 0, 1, 5, 2, 2, 2, 5, 0, 2, 5, 3, 1, 0, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 5, 5, 1, 2, 0, 3, 3, 1, 5, 5, 4, 1, 5, 0, 1, 2, 1, 2, 4, 1, 2, 0, 0, 4, 1, 1, 2, 5, 2, 3, 3, 2, 0, 2, 2, 1, 2, 0, 3, 1, 1, 0, 0, 1, 4, 5, 5, 2, 4, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 5, 2, 2, 2, 2, 0, 2, 2, 0, 2, 1, 3, 2, 2, 0, 5, 2, 2, 0, 2, 3, 2, 5, 5]

        # for subject, csv_path in subject_csv_map.items():
        #     df = pd.read_csv(csv_path)
            
        #     # 筛选 train/test 文件名
        #     train_files = list(df[df.iloc[:,0] == 'train'].iloc[:,5].values)
        #     test_files  = list(df[df.iloc[:,0] == 'test'].iloc[:,5].values)
            
        #     # 假设 a 是 test_files 对应的值列表
        #     # 如果你有 train_files 对应的 a_train，也可以同样操作
        #     # 这里以 test_files 为例
        #     # 长度必须一致
        #     assert len(test_files) == len(a), "test_files 和 a 长度不一致"
            
        #     # 得到 a 排序后的索引
        #     a_sorted_indices = sorted(range(len(a)), key=lambda i: a[i])
            
        #     # 使用这个索引重新排列 test_files
        #     test_files_sorted = [test_files[i] for i in a_sorted_indices]
            
        #     # 保存结果
        #     subject_file_set[subject] = {
        #         'train': train_files,       # train 不排序
        #         'test' : test_files_sorted  # test 按 a 排序
        #     }
        directory = self.img_directory
        
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        img_directory = self.img_directory
        dep_directory = self.img_depth
        seg_directory = self.img_segment
        # if self.classes is not None:
        #     dirnames = [dirnames[i] for i in self.classes]

        #     img_directory = self.img_directory
        #     dep_directory = self.img_depth
        #     seg_directory = self.img_segment
        # all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        # all_folders.sort()  
        new_texts = []
        if self.classes is not None and self.pictures is not None:
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []  

            folder_path = os.path.join(img_directory)
            seg_path = os.path.join(seg_directory)
            dep_path = os.path.join(dep_directory)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            seg_images = [img for img in os.listdir(seg_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # dep_images = [all_images for img in os.listdir(dep_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            seg_images.sort()
            # dep_images.sort()
            # 确保三类图像数量一致（避免索引越界）

            if self.train:
            # 组合路径：每个元素为(原图路径, 分割图路径, 深度图路径)
                for subj in self.subjects:
                    if subj == self.exclude_subject:  
                        continue
                    for i in range(len(all_images)):
                        img_name = all_images[i]

                        # 判断图片是否属于当前 subject 的 train/test
                        
                        if img_name not in subject_file_set[subj]['train' if self.train else 'test']:
                            continue  # 不属于当前 train/test，跳过

                        img_path     = os.path.join(folder_path, img_name)
                        seg_img_path = os.path.join(seg_path, seg_images[i])
                        dep_img_path = os.path.join(dep_path, img_name)

                    # 将三个路径作为元组添加到 images
                        images.append( (img_path, seg_img_path, dep_img_path) )
                        new_texts.append(texts[i])  
            else:
                for subj in self.subjects:
                    if subj != self.exclude_subject and self.exclude_subject is not None:  
                        continue
                    for i in range(len(all_images)):
                        img_name = all_images[i]

                        # 判断图片是否属于当前 subject 的 train/test
                        
                        if img_name not in subject_file_set[subj]['train' if self.train else 'test']:
                            continue  # 不属于当前 train/test，跳过

                        img_path     = os.path.join(folder_path, img_name)
                        seg_img_path = os.path.join(seg_path, seg_images[i])
                        dep_img_path = os.path.join(dep_path, img_name)

                    # 将三个路径作为元组添加到 images
                        images.append( (img_path, seg_img_path, dep_img_path) )
                        new_texts.append(texts[i])  

        else:
            
            print("Error")
        texts = new_texts       
        print("self.subjects", self.subjects)
        print("exclude_subject", self.exclude_subject)
        # for subject in self.subjects:
        #     if self.train:
        #         if subject == self.exclude_subject:  
        #             continue            
        #         # print("subject:", subject)    
        #         file_name = 'train_responses.pkl'

        #         file_path = os.path.join(self.data_path, subject, file_name)
        #         data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        #         preprocessed_eeg_data = torch.from_numpy(data['eeg']).float().detach()                
        #         # times = torch.from_numpy(data['times']).detach()[50:]
        #         # ch_names = data['ch_names']  

        #         n_classes = 720
        #         samples_per_class = 12
                
        #         if self.classes is not None and self.pictures is not None:
        #             for c, p in zip(self.classes, self.pictures):
        #                 start_index = c * 1 + p
        #                 if start_index < len(preprocessed_eeg_data):  
        #                     preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]  
        #                     labels = torch.full((1,), c, dtype=torch.long).detach()  
        #                     data_list.append(preprocessed_eeg_data_class)
        #                     label_list.append(labels)  

        #         elif self.classes is not None and self.pictures is None:
        #             for c in self.classes:
        #                 start_index = c * samples_per_class
        #                 preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
        #                 labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()  
        #                 data_list.append(preprocessed_eeg_data_class)
        #                 label_list.append(labels)

        #         else:
        #             for i in range(n_classes):
        #                 start_index = i * samples_per_class
        #                 # if self.exclude_subject==None:
        #                 #     preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
        #                 # else:
        #                 preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
        #                 # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
        #                 # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 1)
        #                 # preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
        #                 # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
        #                 labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  
        #                 data_list.append(preprocessed_eeg_data_class)
        #                 label_list.append(labels)

                 
        #     else:
        #         if subject == self.exclude_subject or self.exclude_subject==None:  
        #             file_name = 'test_responses.pkl'
        #             file_path = os.path.join(self.data_path, subject, file_name)
        #             data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        #             preprocessed_eeg_data = torch.from_numpy(data['eeg']).float().detach()         
        #             # times = torch.from_numpy(data['times']).detach()[50:]
        #             # ch_names = data['ch_names']  
        #             n_classes = 100  # Each class contains 1 images
                    
        #             samples_per_class = 1  

        #             for i in range(n_classes):
        #                 if self.classes is not None and i not in self.classes:  # If we've defined specific classes and the current class is not in the list, skip
        #                     continue
        #                 start_index = i * samples_per_class  # Update start_index for each class
        #                 preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
        #                 # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
        #                 labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
        #                 preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
        #                 # print("preprocessed_eeg_data_class", preprocessed_eeg_data_class.shape)
        #                 data_list.append(preprocessed_eeg_data_class)
        #                 label_list.append(labels)  # Add labels to the label list
        #         else:
        #             continue

        # 预先读取 CSV，生成每个 subject 的图片排序


        # 循环 subjects
        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:  
                    continue
                file_name = 'train_responses.pkl'
                file_path = os.path.join(self.data_path, subject, file_name)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                preprocessed_eeg_data = torch.from_numpy(data).float().detach()

                # 展平成 [8640, 6036]
                # preprocessed_eeg_data = preprocessed_eeg_data.view(-1, preprocessed_eeg_data.size(-1))

                # 排序
                # sort_idx = subject_file_order[subject]['train']  # 长度应该是 8640
                # preprocessed_eeg_data = preprocessed_eeg_data[sort_idx]

                preprocessed_eeg_data = preprocessed_eeg_data.view(720 * 12, -1)
                
                n_classes, samples_per_class = 720, 12
                for i in range(720):
                    start_index = i * samples_per_class
                    preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + samples_per_class]
                    labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                    data_list.append(preprocessed_eeg_data_class)
                    label_list.append(labels)
                    
            else:  # test
                if subject == self.exclude_subject or self.exclude_subject==None:  
                    file_name = 'test_responses.pkl'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    print(file_path)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    preprocessed_eeg_data = torch.from_numpy(data).float().detach()  # [100,12,6036]
                    
                    # 展平成 [8640, 6036]
                    # preprocessed_eeg_data = preprocessed_eeg_data.view(-1, preprocessed_eeg_data.size(-1))

                    # # 排序
                    # sort_idx = subject_file_order[subject]['test']  # 长度应该是 8640
                    # preprocessed_eeg_data = preprocessed_eeg_data[sort_idx]

                    # # 如果还需要 [720, 12, 6036] 形状，就 reshape 回去
                    # preprocessed_eeg_data = preprocessed_eeg_data.view(100, 12, -1)
                    
                    for i in range(100):
                        start_index = i
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+1]  # [1,12,6036]
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), dim=0)
                        labels = torch.full((1,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                else:
                    continue
        # datalist: (subjects * classes) * (10 * 4 * 17 * 100)
        # data_tensor: (subjects * classes * 10 * 4) * 17 * 100
        # data_list = np.mean(data_list, )
        # print("data_list", len(data_list))
        if self.train:
        # print("data_list", *data_list[0].shape[1:])            
        # 假设 target_len = 7000
            target_len = 7000

            padded_list = []
            for x in data_list:
                # x 的形状是 [1, seq_len]
                # print(x.shape)
                seq_len = x.shape[1]
                if seq_len < target_len:
                    # 在最后补零
                    pad_size = target_len - seq_len
                    padded = torch.cat([x, torch.zeros((x.shape[0], pad_size), dtype=x.dtype, device=x.device)], dim=1)
                else:
                    padded = x[:, :target_len]  # 如果超过7000，截断
                padded_list.append(padded)

            # 拼接成 [N, 7000]
            data_tensor = torch.cat(padded_list, dim=0)   
            print("data_tensor", data_tensor.shape)
        else:           
            target_len = 7000

            padded_list = []
            for x in data_list:
                # x 的形状是 [1, seq_len]
                seq_len = x.shape[0]
                if seq_len < target_len:
                    # 在最后补零
                    pad_size = target_len - seq_len
                    padded = torch.cat([x, torch.zeros((pad_size), dtype=x.dtype, device=x.device)], dim=0)
                else:
                    padded = x[:, :target_len]  # 如果超过7000，截断
                padded_list.append(padded)

            # 拼接成 [N, 7000]
            data_tensor = torch.cat(padded_list, dim=0)   
            data_tensor = data_tensor.view(100, -1)
            # label_tensor = torch.cat(label_list, dim=0)
            # print("label_tensor", label_tensor.shape)
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
        # print("data_tensor", data_tensor.shape)
        # label_list: (subjects * classes) * 10
        # label_tensor: (subjects * classes * 10)
        # print("label_tensor = torch.cat(label_list, dim=0)")
        # print(label_list)
        label_tensor = torch.cat(label_list, dim=0)
        # label_tensor = torch.cat(label_list, dim=0)
        # print(label_tensor[:300])
        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            # label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)        
                mapping = {val.item(): index for index, val in enumerate(unique_values)}   
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        else:
            # label_tensor = label_tensor.repeat_interleave(80)
            # if self.classes is not None:
            #     unique_values = torch.unique(label_tensor, sorted=False)
           
            #     mapping = {val.item(): index for index, val in enumerate(torch.flip(unique_values, [0]))}
            #     label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
            pass      

                    
        # self.times = times
        # self.ch_names = ch_namesC

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        # print("self.times", self.times.shape)
        # print("indices", indices)
        # print("indices", indices.shape)
        # print("eeg_data", eeg_data.shape)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]
        # print(f"extracted_data shape: {extracted_data.shape}")

        return extracted_data
    

    def Textencoder(self, text):   
        # 输入维度 为列表 N, 4
        batch_size = 32  # number of text *groups*
        text_features_list = []

        # assert len(text) % 4 == 0, "Total number of texts should be a multiple of 4"
        grouped_text = [text[i:i+1][0][0] for i in range(0, len(text))]  # 每组4句话
        total_groups = len(grouped_text)

        for i in range(0, total_groups, batch_size):
            batch_groups = grouped_text[i:i + batch_size]  # 取 batch_size 个 group
            flat_texts = [sentence for group in batch_groups for sentence in group]  # 展平为一维
            text_inputs = torch.cat([open_clip.tokenize(t) for t in flat_texts]).to(device)

            with torch.no_grad():
                text_features = vlmodel.encode_text(text_inputs)

            text_features = F.normalize(text_features, dim=-1).detach()
            text_features = text_features.view(-1, 4, 1024)  # [batch_size, 4, 1024]
            text_features_list.append(text_features.cpu()) 

        all_text_features = torch.cat(text_features_list, dim=0)  # [total_groups, 4, 1024]

        print(f"Text features shape: {all_text_features.shape}")
        return all_text_features

        
        
    # def ImageEncoder(self,images):
    #     batch_size = 20  
    #     image_features_list = []
      
    #     for i in range(0, len(images), batch_size):
    #         batch_images = images[i:i + batch_size]
    #         image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

    #         with torch.no_grad():
    #             batch_image_features = vlmodel.encode_image(image_inputs) # torch.Size([20, 1024]) torch.Size([20, 3, 224, 224])
    #             batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

    #         image_features_list.append(batch_image_features)

    #     image_features = torch.cat(image_features_list, dim=0)
        
    #     return image_features
    def ImageEncoder(self, images):
        """
        images: [B, 3, 500, 500]
        depth: [B, 3, 500, 500]
        mask:  [B, 1, 500, 500]
        """

        
        batch_size = 4
        image_features_list = []
        for i in range(0, len(images), batch_size):

            image1 = images[i:i + batch_size]
            image = [tup[0] for tup in image1]
            depth = [tup[2] for tup in image1]
            mask = [tup[1] for tup in image1]
            
            image = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in image]).to(device)
            depth = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in depth]).to(device)
            mask = torch.stack([preprocess_mask(img) for img in mask]).to(device) 
            # 图像预处理后提取原始视觉embedding
            masked_images = image * mask
            image_inputs = F.interpolate(masked_images, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)  # [B, 1024]
                batch_image_features_po = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)

            inverted_masks = 1 - mask  # 关键修改：通过1减去原mask实现取反
            
            # 2. 应用取反后的mask
            masked_images = image * inverted_masks  # [B, 3, H, W]，原0区域保留，1区域屏蔽
            image_inputs = F.interpolate(masked_images, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)  # [B, 1024]
                batch_image_features_ba = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
            # 纹理特征提取


            with torch.no_grad():
                resnet_feats = self.resnet_layer4(image)  # [B, 2048, H/32, W/32] → [B, 2048, 16, 16]
                global_feats = torch.mean(resnet_feats, dim=[2, 3], keepdim=True)
                texture_feat = global_feats[:, :1024] + global_feats[:, 1024:]  # [B, 1024]  # 无参数操作
                texture_feat = texture_feat.squeeze()
                texture_feat = texture_feat / texture_feat.norm(dim=-1, keepdim=True)
            # 深度图映射
            depth = 0.299 * depth[:, 0:1, :, :] + 0.587 * depth[:, 1:2, :, :] + 0.114 * depth[:, 2:3, :, :]  # [B, 1, H, W]
            depth_down = F.adaptive_avg_pool2d(depth, (32, 32))  # [B, 1, 16, 16]
            depth_flat = depth_down.view(batch_size, -1)  # [B, 3*16*16]
            # depth_feat = F.linear(depth_flat, depth_linear_weight)  # [B, 1024]
            depth_feat = depth_flat / depth_flat.norm(dim=-1, keepdim=True)

            # images图映射
            images_down = F.adaptive_avg_pool2d(image, (32, 32))  # [B, 1, 16, 16]
            images_flat = images_down.view(batch_size, -1, 1024)  # [B, 1*16*16]
            images_flat = images_flat[:, 0, :] + images_flat[:, 1, :] + images_flat[:, 2, :]
            # images_feat = F.linear(images_flat, images_linear_weight)  # [B, 1024]
            images_feat = images_flat / images_flat.norm(dim=-1, keepdim=True)

            # 拼接四个模态
            image_features = torch.stack([batch_image_features_po, batch_image_features_ba, depth_feat, images_feat, texture_feat], dim=1)  # [B, 4, 1024]
            image_features_list.append(image_features)
        image_features = torch.cat(image_features_list, dim=0)
        return image_features
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]
        
        # if self.pictures is None:
        #     if self.classes is None:
        #         index_n_sub_train = self.n_cls * 10 * 4
        #         index_n_sub_test = self.n_cls * 1 * 80
        #     else:
        #         index_n_sub_test = len(self.classes)* 1 * 80
        #         index_n_sub_train = len(self.classes)* 10 * 4
        #     # text_index: classes
        #     if self.train:
        #         text_index = (index % index_n_sub_train) // (10 * 4)
        #     else:
        #         text_index = (index % index_n_sub_test)
        #     # img_index: classes * 10
        #     if self.train:
        #         img_index = (index % index_n_sub_train) // (4)
        #     else:
        #         img_index = (index % index_n_sub_test)
        # else:
        #     if self.classes is None:
        #         index_n_sub_train = self.n_cls * 1 * 4
        #         index_n_sub_test = self.n_cls * 1 * 80
        #     else:
        #         index_n_sub_test = len(self.classes)* 1 * 80
        #         index_n_sub_train = len(self.classes)* 1 * 4
        #     # text_index: classes
        #     if self.train:
        #         text_index = (index % index_n_sub_train) // (1 * 4)
        #     else:
        #         text_index = (index % index_n_sub_test)
        #     # img_index: classes * 10
        #     if self.train:
        #         img_index = (index % index_n_sub_train) // (4)
        #     else:
        #         img_index = (index % index_n_sub_test)


        # Calculate text and image indices



        index_n_sub_train = 8640
        index_n_sub_test = 100

        if self.train:
            text_index = (index % index_n_sub_train) // (1)
            img_index = (index % index_n_sub_train) // (1)
        else:
            text_index = (index % index_n_sub_test) // (1)
            img_index = (index % index_n_sub_test) // (1)


        text = self.text[text_index]
        img = self.img[img_index]
        
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        
        return x, label, text, text_features, img, img_features
        # """Get item by index"""
        """Get item by index with subject-specific handling"""
        # Find which subject this index belongs to
        # subject_idx = None
        # for i, cum_len in enumerate(self.cumulative_data_lens[1:]):
        #     if index < cum_len:
        #         subject_idx = i
        #         break
        # subject_offset = index - self.cumulative_data_lens[subject_idx]
        
        # Get data and label





        # x = self.data[subject_idx][subject_offset]
        # label = self.labels[subject_idx][subject_offset]
        # subject_id = self.subjects[subject_idx]  # Get subject identifier
        
        # # Pad fMRI data to fixed length
        # target_length = 7000
        # if x.shape[0] < target_length:
        #     padding_size = target_length - x.shape[0]
        #     x = F.pad(x, (0, padding_size), value=0)
        # elif x.shape[0] > target_length:
        #     x = x[:target_length]

        # # Calculate text and image indices
        # index_n_sub_train = self.n_cls * 12 * 1
        # index_n_sub_test = self.n_cls * 12 * 1

        # if self.train:
        #     text_index = (subject_offset % index_n_sub_train) // (12 * 1)
        #     img_index = (subject_offset % index_n_sub_train) // (1)
        # else:
        #     text_index = (subject_offset % index_n_sub_test) // (1)
        #     img_index = (subject_offset % index_n_sub_test) // (1)
        
        # # Get text, image and features
        # text = self.text[text_index]
        # img = self.img[img_index]
        # text_features = self.text_features[text_index]  

        # img_features = self.img_features[img_index]
   
            
        # return x, label, text, text_features, img, img_features

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__":
    data_path = data_path
    train_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=True)    
    test_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=False)
    
    
    
    
    # 100 Hz
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    i = 80*1-1
    x, label, text, text_features, img, img_features  = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            
    
        