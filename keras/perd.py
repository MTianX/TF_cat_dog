import keras
import numpy as np
import pandas as pd
import os
from data import create_dir_and_data

#测试图片根目录
test_path = r'D:\study\T\test'


def load_img(filename):
    #读取图片，修改图片大小并进行归一化，将其值返回
    img = keras.preprocessing.image.load_img(filename,target_size=(150,150))
    img_tensor = keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor /= 255.
    return img_tensor


file_names = ['{}.jpg'.format(index) for index in range(1,12501)]
y_list = []
model = keras.models.load_model('model.h5')
for filename in file_names[:12500]:
    print('正在预测{}文件'.format(filename))
    path = os.path.join(test_path,filename)
    y = model.predict(load_img(path))
    y_list.append(y)

print('正在写入csv文件')
answer = pd.read_csv(open('sample_submission.csv'))
answer['label'][:12500] = y_list[:12500]
print(answer.head())
answer.to_csv('submission.csv',index=False)