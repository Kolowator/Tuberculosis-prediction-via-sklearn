# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:06:22 2024

@author: Admin
"""

import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def get_min_dimensions(image_paths):
       min_width, min_height = float('inf'), float('inf')
       for image_path in image_paths:
           with Image.open(image_path) as img:
               width, height = img.size
               if width < min_width:
                   min_width = width
               if height < min_height:
                   min_height = height
       return min_width, min_height

def get_max_dimensions(image_paths):
       max_width, max_height = float(0), float(0)
       for image_path in image_paths:
           with Image.open(image_path) as img:
               width, height = img.size
               if width > max_width:
                   max_width = width
               if height > max_height:
                   max_height = height
       return max_width, max_height    

def resize_images(image_paths, target_size, output_folder):
       if not os.path.exists(output_folder):
           os.makedirs(output_folder)
       
       for image_path in image_paths:
           with Image.open(image_path) as img:
               resized_img = img.resize(target_size, Image.LANCZOS)
               base_name = os.path.basename(image_path)
               resized_img.save(os.path.join(output_folder, base_name))    

image_dir = 'image'
mask_dir = 'mask'
maskedim_dir = 'maskedim'
imagedata = []
maskdata = []

image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

# assert len(image_files) == len(mask_files), "Количество изображений и масок должно совпадать"

# for image_file, mask_file in zip(image_files, mask_files):
#     # Загружаем изображение и маску
#     image = cv2.imread(os.path.join(image_dir, image_file))
#     mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
    
#     binary_mask = mask / 255
    
#     masked_image = cv2.bitwise_and(image, image, mask=mask)
    
#     vertical_sum = np.sum(masked_image, axis=0)
#     non_zero_indices = np.where(vertical_sum > 0)[0]
#     left_bound = non_zero_indices[0]
#     right_bound = non_zero_indices[-1]
    
#     horizontal_sum = np.sum(masked_image, axis=1)
#     non_zero_indices2 = np.where(horizontal_sum > 0)[0]
#     top_bound = non_zero_indices2[0]
#     bottom_bound = non_zero_indices2[-1]
    
#     cropped_image = masked_image[top_bound:bottom_bound+1, left_bound:right_bound+1]
    
#     masked_file = os.path.join(maskedim_dir, image_file)
#     cv2.imwrite(masked_file, cropped_image)

# img = mpimg.imread(f"{image_dir}/{image_files[0]}" )
# plt.imshow(img)
# plt.title('Оригинальное изображение ФЛГ')
# plt.axis('off')  # Отключение осей
# plt.show()

# msk = mpimg.imread(f"{mask_dir}/{mask_files[0]}" )
# plt.imshow(msk,cmap='gray')
# plt.title('Маска для ФЛГ')
# plt.axis('off')  # Отключение осей
# plt.show()


masked_files = [os.path.join(maskedim_dir, f) for f in os.listdir(maskedim_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

# mskd = mpimg.imread(f"{masked_files[0]}" )
# plt.imshow(mskd,cmap='gray')
# plt.title('Изображение ФЛГ с наложеной маской')
# plt.axis('off')  # Отключение осей
# plt.show()

# min_width, min_height = get_min_dimensions(masked_files)
# print(f"Минимальные размеры: {min_width}x{min_height}")

# max_width, max_height = get_max_dimensions(masked_files)
# print(f"Минимальные размеры: {max_width}x{max_height}")

resized = 'resizedmaskedim'
# resize_images(masked_files, (min_width, min_height),resized)

Meta = pd.read_csv('MetaData.csv')

data = []

for imagename in os.listdir(resized):
    # Полный путь к изображению
    image_path = os.path.join(resized, imagename)
    image = Image.open(image_path)
    image_array = np.array(image).flatten()
    
    label = Meta.loc[Meta['id'] == int(imagename[:-4]),'ptb'].values[0]
    data.append({'label': label, 'image_data': image_array})


df = pd.DataFrame(data)

df.to_csv('output.csv', index=False, encoding='utf-8')

X = np.stack(df['image_data'].values)
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

model = RandomForestClassifier(n_estimators=100,max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

svcm = SVC(C=2.0)
svcm.fit(X_train, y_train)
y_pred = svcm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")