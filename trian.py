import tensorflow as tf
import numpy as np
import os
import ops
import Dataset

N_CLASSES = 2  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
BATCH_SIZE = 100  #每批数据的大小
MAX_STEP = 10000 # 训练的步数
learning_rate = 0.0001 # 学习率
channel_num =3
logs_train_dir = r'D:\study\T\MachineLearn\cat_dog\model'

def load_data(class_lists, tag_lists,rate = 0.8):
    list_len = len(tag_lists)
    class_len = len(class_lists)
    if list_len != class_len:
        print('特征列表与标签列表长度不匹配')
        return None
    index = int(list_len*rate)
    train_class_list = np.array(class_lists[:index],dtype=np.string_)
    train_tag_list = np.array(tag_lists[:index],dtype=np.int32)
    test_class_list = np.array(class_lists[index:],dtype=np.string_)
    test_tag_list = np.array(tag_lists[index:],dtype=np.int32)
    print('测试集数量为{}，验证集数量为{}'.format(len(train_tag_list),len(test_tag_list)))
    return train_class_list,train_tag_list,test_class_list,test_tag_list

def run_training():
    sess = tf.Session()
    class_lists, tag_lists = Dataset.Create_class_list(list_rate=0.5)
    class_lists, tag_lists = Dataset.shuffle_data(class_lists, tag_lists,10)
    train_class_list, train_tag_list, test_class_list, test_tag_list = load_data(class_lists, tag_lists)
    x,y = tf.placeholder(dtype=tf.string,shape=None),tf.placeholder(dtype=tf.int32,shape=[None])
    batch_size = tf.placeholder(dtype=tf.int64)
    y_one_hot = tf.one_hot(y,depth=N_CLASSES)
    keep_prob = tf.placeholder(dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices((x,y_one_hot)).map(Dataset._parse_function,num_parallel_calls=4).batch(batch_size).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x,y_one_hot)).map(Dataset._parse_function,num_parallel_calls=4).batch(batch_size)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    features,labels = iter.get_next()
    print(features,labels)

    train_init_op = iter.make_initializer(train_dataset)
    test_init_op = iter.make_initializer(test_dataset)

    train_logits = ops.inference(features,N_CLASSES,keep_prob)
    train_loss = ops.losses(train_logits,labels)
    train_op = ops.trainning(train_loss,learning_rate)
    train_acc = ops.evaluation(train_logits,labels)
    summary_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op,feed_dict={x:train_class_list, y:train_tag_list, batch_size:BATCH_SIZE})
    testfg = False
    for step in range(1,MAX_STEP):
            try:
                if testfg == True:
                    testfg = False
                    sess.run(train_init_op, feed_dict={x: train_class_list, y: train_tag_list, batch_size: BATCH_SIZE})
                _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc],feed_dict={keep_prob:0.5})
                if step%5 ==0:
                    print('Step %d,train_loss = %.3f,train accuracy = %.3f'%(step,tra_loss,tra_acc))
                    summary_str = sess.run(summary_op,feed_dict={keep_prob:0.5})
                    train_writer.add_summary(summary_str,step)
                if step%100 == 0:
                    testfg = True
                    sess.run(test_init_op,
                             feed_dict={x: test_class_list, y: test_tag_list, batch_size: len(test_tag_list)})
                    print('test acc:{:4f}'.format(sess.run(train_acc,feed_dict={keep_prob:1.0})))
                if step%1000 == 0 or (step +1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                sess.run(iter.initializer, feed_dict={x: train_class_list, y: train_tag_list, batch_size: BATCH_SIZE})


if __name__ == '__main__':
    run_training()