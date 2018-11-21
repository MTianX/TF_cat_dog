import os
import shutil

#猫狗数据集路径
data_dir = r'D:\study\T\train'
#储存较小数据目录
base_dir = r'D:\study\T\MachineLearn\keras\cat_dog\cat_and_dog_small'

def create_dir(pathlist):
    for path in pathlist:
        if os.path.isdir(path):
            print('所创建的文件夹已存在，路径为{}'.format(path))
            continue
        os.mkdir(path)
        print('已创建文件夹，路径为{}'.format(path))

def copy2dir(fname_list,src_dir,dst_dir):
    if os.path.isdir(src_dir) is False:
        os.mkdir(src_dir)
    if os.path.isdir(dst_dir) is False:
        os.mkdir(dst_dir)

    for fname in fname_list:
        src = os.path.join(src_dir,fname)
        dst = os.path.join(dst_dir,fname)
        if os.path.isfile(dst):
            print('{} 文件已存在'.format(fname))
            continue
        shutil.copyfile(src, dst)

def create_dir_and_data():
    #划分训练集、验证集和测试集目录
    train_dir = os.path.join(base_dir,'train')
    validation_dir = os.path.join(base_dir,'validation')
    tess_dir = os.path.join(base_dir,'test')
    #创建目录
    # create_dir([train_dir,validation_dir,tess_dir])
    #猫、狗训练目录
    train_cats_dir = os.path.join(train_dir,'cats')
    train_dogs_dir = os.path.join(train_dir,'dogs')
    #猫、狗测试目录
    test_cats_dir = os.path.join(tess_dir,'cats')
    test_dogs_dir = os.path.join(tess_dir,'dogs')
    #猫、狗验证目录
    validation_cats_dir = os.path.join(validation_dir,'cats')
    validation_dogs_dir = os.path.join(validation_dir,'dogs')
    # #创建目录
    # create_dir([train_cats_dir,train_dogs_dir,test_cats_dir,test_dogs_dir,validation_cats_dir,validation_dogs_dir])
    #
    # fnames_cat_1 = ['cat.{}.jpg'.format(index) for index in range(1000)]
    # fnames_cat_2 = ['cat.{}.jpg'.format(index) for index in range(1000,1500)]
    # fnames_cat_3 = ['cat.{}.jpg'.format(index) for index in range(1500,2000)]
    #
    # fnames_dog_1 = ['dog.{}.jpg'.format(index) for index in range(1000)]
    # fnames_dog_2 = ['dog.{}.jpg'.format(index) for index in range(1000,1500)]
    # fnames_dog_3 = ['dog.{}.jpg'.format(index) for index in range(1500,2000)]
    #
    # copy2dir(fnames_cat_1,data_dir,train_cats_dir)
    # copy2dir(fnames_cat_2,data_dir,validation_cats_dir)
    # copy2dir(fnames_cat_3,data_dir,test_cats_dir)
    #
    # copy2dir(fnames_dog_1,data_dir,train_dogs_dir)
    # copy2dir(fnames_dog_2,data_dir,validation_dogs_dir)
    # copy2dir(fnames_dog_3,data_dir,test_dogs_dir)
    #
    # print('猫的训练集、验证集和测试集数量分别为：{}、{}、{}'.format(len(os.listdir(train_cats_dir)),
    #       len(os.listdir(validation_cats_dir)),
    #       len(os.listdir(test_cats_dir))))
    # print('狗的训练集、验证集和测试集数量分别为：{}、{}、{}'.format(len(os.listdir(train_dogs_dir)),
    #       len(os.listdir(validation_dogs_dir)),
    #       len(os.listdir(test_dogs_dir))))
    return train_dir,validation_dir,tess_dir

if __name__ == '__main__':
    train_dir, validation_dir, test_dir=create_dir_and_data()
