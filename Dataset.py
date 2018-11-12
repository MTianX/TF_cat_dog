import os
import re
import random
import tensorflow as tf


img_size = 24

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string,channels=3)
  image_float = tf.image.convert_image_dtype(image_decoded,tf.float32)

  label = tf.cast(label,tf.int32)
  image_float.set_shape([None,None,3])
  image_resized = tf.image.resize_images(image_float, [img_size, img_size])
  return image_resized, label

def Create_class_list(list_rate = 0.1):
    rootdir = r'D:\study\T\train'
    cls = ['cat','dog']
    catlist = []
    doglist = []
    cat_tag = []
    dog_tag = []
    list = os.listdir(rootdir)
    for i in range(0,len(list)):
        #cat
        if re.match(cls[0],list[i]):
            catlist.append(rootdir+'\\'+list[i])
            cat_tag.append(0)
        #dog
        elif re.match(cls[1],list[i]):
            doglist.append(rootdir+'\\'+list[i])
            dog_tag.append(1)

    list_index = int(len(cat_tag)*list_rate)

    class_list = catlist[:list_index]+doglist[:list_index]
    tag_list = cat_tag[:list_index]+dog_tag[:list_index]
    print("图片数量为{},标签数量为{}".format(len(class_list),len(tag_list)))
    return class_list,tag_list

def shuffle_data(dataset,tag,randseed):
    random.seed(randseed)
    random.shuffle(dataset)
    random.seed(randseed)
    random.shuffle(tag)
    return dataset, tag

if __name__ == '__main__':
    class_list, tag_list= Create_class_list()
    class_list, tag_list = shuffle_data(class_list, tag_list,10)

