import tensorflow as tf

def get_weights(name,shape,stddev):
    return tf.get_variable(name=name,shape=shape,initializer=tf.truncated_normal_initializer(stddev=stddev),dtype=tf.float32)
def get_biases(name,shape):
    return tf.get_variable(name=name,shape=shape,initializer=tf.constant_initializer(0.1),dtype=tf.float32)

def get_conv2_layer(inputs,filter_size_conv,filter_conv_num,ues_pool = False):

    #卷积
    # weights = get_weights('weights',shape=[filter_size_conv,filter_size_conv,channel_num,filter_conv_num],stddev=0.1)
    # biases = get_biases('biases',shape=[filter_conv_num])
    # conv2 = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
    # conv2 = tf.nn.bias_add(conv2,biases)
    # conv2 = tf.nn.relu(conv2)
    conv2 = tf.layers.conv2d(inputs=inputs,
                             filters=filter_conv_num,
                             kernel_size=filter_size_conv,
                             strides=1,padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.constant_initializer(0.1))
    #池化
    if ues_pool:
        conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=3, strides=(2, 2), padding='same')
        conv2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75,)
    return conv2

def inference(inputs,n_class,keep_prob):
    filter_size = 5
    filter_num_conv1 = 64
    filter_num_conv2 = 64
    filter_num_conv3 = 128


    fc1_layer_size = 384
    fc2_layer_size = 192

    with tf.variable_scope('conv1') as scope:
        conv1 = get_conv2_layer(inputs, filter_size, filter_num_conv1)
        print('conv1 = {}'.format(conv1))
    with tf.variable_scope('conv2') as scope:
        conv2 = get_conv2_layer(conv1, filter_size, filter_num_conv2,ues_pool =True)
        print('conv2 = {}'.format(conv2))
    with tf.variable_scope('conv3') as scope:
        conv3 = get_conv2_layer(conv2, filter_size, filter_num_conv3,ues_pool =True)
        print('conv3 = {}'.format(conv2))


    with tf.variable_scope('fc1') as scope:
        fc1_flatten = tf.layers.flatten(conv3)
        print('fc1_flatten = {}'.format(fc1_flatten))
        dim = fc1_flatten.get_shape()[1]
        fc1 = tf.layers.dense(inputs=fc1_flatten,units=fc1_layer_size,activation=tf.nn.relu,kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),bias_initializer=tf.constant_initializer(0.1))
        print('fc1 = {}'.format(fc1))
        fc1_Dropout = tf.nn.dropout(fc1,keep_prob)
    with tf.variable_scope('fc2') as scope:
        fc2 = tf.layers.dense(inputs=fc1_Dropout, units=fc2_layer_size, activation=tf.nn.relu,kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),bias_initializer=tf.constant_initializer(0.1))
        print('fc2 = {}'.format(fc2))
        fc2_Dropout = tf.nn.dropout(fc2, keep_prob)
    with tf.variable_scope('softmax') as scope:
        output = tf.layers.dense(inputs=fc2_Dropout,units=n_class,kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),bias_initializer=tf.constant_initializer(0.1))
    return output

def losses(logits,labels):
    with tf.variable_scope('loss') as scope:
        loss = tf.losses.softmax_cross_entropy(logits = logits,onehot_labels = labels)
    tf.summary.scalar(scope.name + '/loss',loss)
    return loss

def trainning(loss,learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0,name = 'global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))
        correct = tf.cast(correct,tf.float32)
        accuracy = tf.reduce_mean(correct)
    tf.summary.scalar(scope.name + '/accuracy',accuracy)
    return accuracy