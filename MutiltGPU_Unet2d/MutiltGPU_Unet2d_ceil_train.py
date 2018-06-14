import tensorflow as tf
import numpy as np
import pandas as pd
import cv2


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, variable_name=None, uniform=True):
    with tf.device('/cpu:0'):
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            initial = tf.random_uniform_initializer(-init_range, init_range)
            return tf.get_variable(name=variable_name, shape=shape, initializer=initial, trainable=True)
        else:
            stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
            initial = tf.truncated_normal_initializer(stddev=stddev)
            return tf.get_variable(name=variable_name, shape=shape, initializer=initial, trainable=True)


# Bias initialization
def bias_variable(shape, variable_name=None):
    with tf.device('/cpu:0'):
        initial = tf.constant_initializer(0.1)
        return tf.get_variable(name=variable_name, shape=shape, initializer=initial, trainable=True)


# 2D convolution
def conv2d(x, W, strides=1):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    return conv_2d


# 2D deconvolution
def deconv2d(x, W, stride=2):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * stride, x_shape[2] * stride, x_shape[3] // stride])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')


# Max Pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Unet crop and concat
def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


def get_cost(Y_pred, Y_gt):
    H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = intersection / denominator
    return loss


# Serve data by batches
def _next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


def _create_conv_net(X, Y, n_class=1):
    inputX = tf.reshape(X, [-1, 512, 512, 1])
    # UNet model
    # layer1->convolution
    # UNet model
    # layer1->convolution
    with tf.variable_scope('layer1'):
        W1_1 = weight_xavier_init(shape=[3, 3, 1, 32], n_inputs=3 * 3 * 1, n_outputs=32,
                                  variable_name='W1_1')
        B1_1 = bias_variable([32], variable_name='B1_1')
        conv1_1 = conv2d(inputX, W1_1) + B1_1
        # conv1_1 = tf.contrib.layers.batch_norm(conv1_1, epsilon=1e-5, scope='bn1')
        conv1_1 = tf.contrib.layers.batch_norm(conv1_1, center=True, scale=True, is_training=phase, scope='bn1')
        conv1_1 = tf.nn.dropout(tf.nn.relu(conv1_1), drop_conv)

    with tf.variable_scope('layer2'):
        W1_2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32, n_outputs=32, variable_name='W1_2')
        B1_2 = bias_variable([32], variable_name='B1_2')
        conv1_2 = conv2d(conv1_1, W1_2) + B1_2
        # conv1_2 = tf.contrib.layers.batch_norm(conv1_2, epsilon=1e-5, scope='bn2')
        conv1_2 = tf.contrib.layers.batch_norm(conv1_2, center=True, scale=True, is_training=phase, scope='bn2')
        conv1_2 = tf.nn.dropout(tf.nn.relu(conv1_2), drop_conv)

    pool1 = max_pool_2x2(conv1_2)
    # layer2->convolution
    with tf.variable_scope('layer3'):
        W2_1 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 32, n_outputs=64, variable_name='W2_1')
        B2_1 = bias_variable([64], variable_name='B2_1')
        conv2_1 = conv2d(pool1, W2_1) + B2_1
        # conv2_1 = tf.contrib.layers.batch_norm(conv2_1, epsilon=1e-5, scope='bn3')
        conv2_1 = tf.contrib.layers.batch_norm(conv2_1, center=True, scale=True, is_training=phase, scope='bn3')
        conv2_1 = tf.nn.dropout(tf.nn.relu(conv2_1), drop_conv)

    with tf.variable_scope('layer4'):
        W2_2 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64, n_outputs=64, variable_name='W2_2')
        B2_2 = bias_variable([64], variable_name='B2_2')
        conv2_2 = conv2d(conv2_1, W2_2) + B2_2
        # conv2_2 = tf.contrib.layers.batch_norm(conv2_2, epsilon=1e-5, scope='bn4')
        conv2_2 = tf.contrib.layers.batch_norm(conv2_2, center=True, scale=True, is_training=phase, scope='bn4')
        conv2_2 = tf.nn.dropout(tf.nn.relu(conv2_2), drop_conv)

    pool2 = max_pool_2x2(conv2_2)

    # layer3->convolution
    with tf.variable_scope('layer5'):
        W3_1 = weight_xavier_init(shape=[3, 3, 64, 128], n_inputs=3 * 3 * 64, n_outputs=128, variable_name='W3_1')
        B3_1 = bias_variable([128], variable_name='B3_1')
        conv3_1 = conv2d(pool2, W3_1) + B3_1
        # conv3_1 = tf.contrib.layers.batch_norm(conv3_1, epsilon=1e-5, scope='bn5')
        conv3_1 = tf.contrib.layers.batch_norm(conv3_1, center=True, scale=True, is_training=phase, scope='bn5')
        conv3_1 = tf.nn.dropout(tf.nn.relu(conv3_1), drop_conv)

    with tf.variable_scope('layer6'):
        W3_2 = weight_xavier_init(shape=[3, 3, 128, 128], n_inputs=3 * 3 * 128, n_outputs=128, variable_name='W3_2')
        B3_2 = bias_variable([128], variable_name='B3_2')
        conv3_2 = conv2d(conv3_1, W3_2) + B3_2
        # conv3_2 = tf.contrib.layers.batch_norm(conv3_2, epsilon=1e-5, scope='bn6')
        conv3_2 = tf.contrib.layers.batch_norm(conv3_2, center=True, scale=True, is_training=phase, scope='bn6')
        conv3_2 = tf.nn.dropout(tf.nn.relu(conv3_2), drop_conv)

    pool3 = max_pool_2x2(conv3_2)

    # layer4->convolution
    with tf.variable_scope('layer7'):
        W4_1 = weight_xavier_init(shape=[3, 3, 128, 256], n_inputs=3 * 3 * 128, n_outputs=256, variable_name='W4_1')
        B4_1 = bias_variable([256], variable_name='B4_1')
        conv4_1 = conv2d(pool3, W4_1) + B4_1
        # conv4_1 = tf.contrib.layers.batch_norm(conv4_1, epsilon=1e-5, scope='bn7')
        conv4_1 = tf.contrib.layers.batch_norm(conv4_1, center=True, scale=True, is_training=phase, scope='bn7')
        conv4_1 = tf.nn.dropout(tf.nn.relu(conv4_1), drop_conv)

    with tf.variable_scope('layer8'):
        W4_2 = weight_xavier_init(shape=[3, 3, 256, 256], n_inputs=3 * 3 * 256, n_outputs=256, variable_name='W4_2')
        B4_2 = bias_variable([256], variable_name='B4_2')
        conv4_2 = conv2d(conv4_1, W4_2) + B4_2
        # conv4_2 = tf.contrib.layers.batch_norm(conv4_2, epsilon=1e-5, scope='bn8')
        conv4_2 = tf.contrib.layers.batch_norm(conv4_2, center=True, scale=True, is_training=phase, scope='bn8')
        conv4_2 = tf.nn.dropout(tf.nn.relu(conv4_2), drop_conv)

    pool4 = max_pool_2x2(conv4_2)

    # layer5->convolution
    with tf.variable_scope('layer9'):
        W5_1 = weight_xavier_init(shape=[3, 3, 256, 512], n_inputs=3 * 3 * 256, n_outputs=512, variable_name='W5_1')
        B5_1 = bias_variable([512], variable_name='B5_1')
        conv5_1 = conv2d(pool4, W5_1) + B5_1
        # conv5_1 = tf.contrib.layers.batch_norm(conv5_1, epsilon=1e-5, scope='bn9')
        conv5_1 = tf.contrib.layers.batch_norm(conv5_1, center=True, scale=True, is_training=phase, scope='bn9')
        conv5_1 = tf.nn.dropout(tf.nn.relu(conv5_1), drop_conv)

    with tf.variable_scope('layer10'):
        W5_2 = weight_xavier_init(shape=[3, 3, 512, 512], n_inputs=3 * 3 * 512, n_outputs=512, variable_name='W5_2')
        B5_2 = bias_variable([512], variable_name='B5_2')
        conv5_2 = conv2d(conv5_1, W5_2) + B5_2
        # conv5_2 = tf.contrib.layers.batch_norm(conv5_2, epsilon=1e-5, scope='bn10')
        conv5_2 = tf.contrib.layers.batch_norm(conv5_2, center=True, scale=True, is_training=phase, scope='bn10')
        conv5_2 = tf.nn.dropout(tf.nn.relu(conv5_2), drop_conv)

    # layer6->deconvolution
    with tf.variable_scope('layer11'):
        W6 = weight_xavier_init(shape=[3, 3, 256, 512], n_inputs=3 * 3 * 512, n_outputs=256, variable_name='W6')
        B6 = bias_variable([256], variable_name='B6')
        dconv1 = tf.nn.relu(deconv2d(conv5_2, W6) + B6)
        dconv_concat1 = crop_and_concat(conv4_2, dconv1)

    with tf.variable_scope('layer12'):
        # layer7->convolution
        W7_1 = weight_xavier_init(shape=[3, 3, 512, 256], n_inputs=3 * 3 * 512, n_outputs=256, variable_name='W7_1')
        B7_1 = bias_variable([256], variable_name='B7_1')
        conv7_1 = conv2d(dconv_concat1, W7_1) + B7_1
        # conv7_1 = tf.contrib.layers.batch_norm(conv7_1, epsilon=1e-5, scope='bn11')
        conv7_1 = tf.contrib.layers.batch_norm(conv7_1, center=True, scale=True, is_training=phase, scope='bn11')
        conv7_1 = tf.nn.dropout(tf.nn.relu(conv7_1), drop_conv)

    with tf.variable_scope('layer13'):
        W7_2 = weight_xavier_init(shape=[3, 3, 256, 256], n_inputs=3 * 3 * 256, n_outputs=256, variable_name='W7_2')
        B7_2 = bias_variable([256], variable_name='B7_2')
        conv7_2 = conv2d(conv7_1, W7_2) + B7_2
        # conv7_2 = tf.contrib.layers.batch_norm(conv7_2, epsilon=1e-5, scope='bn12')
        conv7_2 = tf.contrib.layers.batch_norm(conv7_2, center=True, scale=True, is_training=phase, scope='bn12')
        conv7_2 = tf.nn.dropout(tf.nn.relu(conv7_2), drop_conv)

    # layer8->deconvolution
    with tf.variable_scope('layer14'):
        W8 = weight_xavier_init(shape=[3, 3, 128, 256], n_inputs=3 * 3 * 256, n_outputs=128, variable_name='W8')
        B8 = bias_variable([128], variable_name='B8')
        dconv2 = tf.nn.relu(deconv2d(conv7_2, W8) + B8)
        dconv_concat2 = crop_and_concat(conv3_2, dconv2)

    # layer9->convolution
    with tf.variable_scope('layer15'):
        W9_1 = weight_xavier_init(shape=[3, 3, 256, 128], n_inputs=3 * 3 * 256, n_outputs=128, variable_name='W9_1')
        B9_1 = bias_variable([128], variable_name='B9_1')
        conv9_1 = conv2d(dconv_concat2, W9_1) + B9_1
        # conv9_1 = tf.contrib.layers.batch_norm(conv9_1, epsilon=1e-5, scope='bn13')
        conv9_1 = tf.contrib.layers.batch_norm(conv9_1, center=True, scale=True, is_training=phase, scope='bn13')
        conv9_1 = tf.nn.dropout(tf.nn.relu(conv9_1), drop_conv)

    with tf.variable_scope('layer16'):
        W9_2 = weight_xavier_init(shape=[3, 3, 128, 128], n_inputs=3 * 3 * 128, n_outputs=128, variable_name='W9_2')
        B9_2 = bias_variable([128], variable_name='B9_2')
        conv9_2 = conv2d(conv9_1, W9_2) + B9_2
        # conv9_2 = tf.contrib.layers.batch_norm(conv9_2, epsilon=1e-5, scope='bn14')
        conv9_2 = tf.contrib.layers.batch_norm(conv9_2, center=True, scale=True, is_training=phase, scope='bn14')
        conv9_2 = tf.nn.dropout(tf.nn.relu(conv9_2), drop_conv)

    # layer10->deconvolution
    with tf.variable_scope('layer17'):
        W10 = weight_xavier_init(shape=[3, 3, 64, 128], n_inputs=3 * 3 * 128, n_outputs=64, variable_name='W10')
        B10 = bias_variable([64], variable_name='B10')
        dconv3 = tf.nn.relu(deconv2d(conv9_2, W10) + B10)
        dconv_concat3 = crop_and_concat(conv2_2, dconv3)

    # layer11->convolution
    with tf.variable_scope('layer18'):
        W11_1 = weight_xavier_init(shape=[3, 3, 128, 64], n_inputs=3 * 3 * 128, n_outputs=64, variable_name='W11_1')
        B11_1 = bias_variable([64], variable_name='B11_1')
        conv11_1 = conv2d(dconv_concat3, W11_1) + B11_1
        # conv11_1 = tf.contrib.layers.batch_norm(conv11_1, epsilon=1e-5, scope='bn15')
        conv11_1 = tf.contrib.layers.batch_norm(conv11_1, center=True, scale=True, is_training=phase, scope='bn15')
        conv11_1 = tf.nn.dropout(tf.nn.relu(conv11_1), drop_conv)

    with tf.variable_scope('layer19'):
        W11_2 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64, n_outputs=64, variable_name='W11_2')
        B11_2 = bias_variable([64], variable_name='B11_2')
        conv11_2 = conv2d(conv11_1, W11_2) + B11_2
        # conv11_2 = tf.contrib.layers.batch_norm(conv11_2, epsilon=1e-5, scope='bn16')
        conv11_2 = tf.contrib.layers.batch_norm(conv11_2, center=True, scale=True, is_training=phase, scope='bn16')
        conv11_2 = tf.nn.dropout(tf.nn.relu(conv11_2), drop_conv)

    with tf.variable_scope('layer20'):
        # layer 12->deconvolution
        W12 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 64, n_outputs=32, variable_name='W12')
        B12 = bias_variable([32], variable_name='B12')
        dconv4 = tf.nn.relu(deconv2d(conv11_2, W12) + B12)
        dconv_concat4 = crop_and_concat(conv1_2, dconv4)

    # layer 13->convolution
    with tf.variable_scope('layer21'):
        W13_1 = weight_xavier_init(shape=[3, 3, 64, 32], n_inputs=3 * 3 * 64, n_outputs=32, variable_name='W13_1')
        B13_1 = bias_variable([32], variable_name='B13_1')
        conv13_1 = conv2d(dconv_concat4, W13_1) + B13_1
        # conv13_1 = tf.contrib.layers.batch_norm(conv13_1, epsilon=1e-5, scope='bn17')
        conv13_1 = tf.contrib.layers.batch_norm(conv13_1, center=True, scale=True, is_training=phase, scope='bn17')
        conv13_1 = tf.nn.dropout(tf.nn.relu(conv13_1), drop_conv)

    with tf.variable_scope('layer22'):
        W13_2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32, n_outputs=32, variable_name='W13_2')
        B13_2 = bias_variable([32], variable_name='B13_2')
        conv13_2 = conv2d(conv13_1, W13_2) + B13_2
        # conv13_2 = tf.contrib.layers.batch_norm(conv13_2, epsilon=1e-5, scope='bn18')
        conv13_2 = tf.contrib.layers.batch_norm(conv13_2, center=True, scale=True, is_training=phase, scope='bn18')
        conv13_2 = tf.nn.dropout(tf.nn.relu(conv13_2), drop_conv)
    # layer14->output
    with tf.variable_scope('layer23'):
        W14 = weight_xavier_init(shape=[1, 1, 32, n_class], n_inputs=1 * 1 * 32, n_outputs=1, variable_name='W14')
        B14 = bias_variable([n_class], variable_name='B14')
        output_map = tf.nn.sigmoid(conv2d(conv13_2, W14) + B14)

        loss = get_cost(Y_pred=output_map, Y_gt=Y)
    return loss, output_map


def _make_parallel(num_gpus, **kwargs):
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)
    return in_splits


csvmaskdata = pd.read_csv('ceil_mask.csv')
csvimagedata = pd.read_csv('ceil_image.csv')
maskdata = csvmaskdata.iloc[:, :].values
imagedata = csvimagedata.iloc[:, :].values
# shuffle imagedata and maskdata together
perm = np.arange(len(csvimagedata))
np.random.shuffle(perm)
imagedata = imagedata[perm]
maskdata = maskdata[perm]
# parameter
learning_rate = 0.0001
train_epochs = 50000
imageHeight = 512
imageWidth = 512
imageChannel = 1
# modify the batch_size and gpu number same time
batch_size = 4
num_gpus=2

with tf.Graph().as_default(), tf.device('/cpu:0'):
    X = tf.placeholder("float", shape=[None, imageHeight, imageWidth, imageChannel])
    Y_gt = tf.placeholder("float", shape=[None, imageHeight, imageWidth, imageChannel])
    lr = tf.placeholder('float')
    phase = tf.placeholder(tf.bool)
    drop_conv = tf.placeholder('float')
    opt = tf.train.AdamOptimizer(learning_rate)

    # make data into every gpu
    out_split = []
    with tf.variable_scope(tf.get_variable_scope()):
        in_splits = _make_parallel(num_gpus, X=X, Y=Y_gt)
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                cost, _ = _create_conv_net(**{k: v[i] for k, v in in_splits.items()})
                tf.get_variable_scope().reuse_variables()
                out_split.append(cost)
        cost = tf.concat(out_split, axis=0)
    cost = -tf.reduce_mean(cost)
    train_op = opt.minimize(cost, colocate_gradients_with_ops=True)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        _, Y_pred = _create_conv_net(X, Y_gt)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.all_variables())
    tf.summary.scalar("loss", cost)
    tf.summary.scalar("accuracy", -cost)
    merged_summary_op = tf.summary.merge_all()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    summary_writer = tf.summary.FileWriter('log/', graph=tf.get_default_graph())
    sess.run(init)

    DISPLAY_STEP = 1
    index_in_epoch = 0
    for i in range(train_epochs):
        # get new batch
        batch_xs_path, batch_ys_path, index_in_epoch = _next_batch(imagedata, maskdata, batch_size, index_in_epoch)
        batch_xs = np.empty((len(batch_xs_path), imageHeight, imageWidth, imageChannel))
        batch_ys = np.empty((len(batch_ys_path), imageHeight, imageWidth, imageChannel))
        for num in range(len(batch_xs_path)):
            image = cv2.imread(batch_xs_path[num][0], cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(batch_ys_path[num][0], cv2.IMREAD_GRAYSCALE)
            batch_xs[num, :, :, :] = np.reshape(image, (imageHeight, imageWidth, imageChannel))
            batch_ys[num, :, :, :] = np.reshape(label, (imageHeight, imageWidth, imageChannel))
        # Normalize from [0:255] => [0.0:1.0]
        batch_xs = batch_xs.astype(np.float)
        batch_ys = batch_ys.astype(np.float)
        batch_xs = np.multiply(batch_xs, 1.0 / 255.0)
        batch_ys = np.multiply(batch_ys, 1.0 / 255.0)
        # check progress on every 1st,2nd,...,10th,20th,...,100th... step
        if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
            train_loss = sess.run(cost, feed_dict={X: batch_xs,
                                                   Y_gt: batch_ys,
                                                   lr: learning_rate,
                                                   phase: 1,
                                                   drop_conv: 0.8})
            print('epochs %d training_loss ,Training_accuracy => %.5f,%.5f ' % (i, train_loss, -train_loss))
            with sess.as_default():
                pred = sess.run(Y_pred, feed_dict={X: batch_xs,
                                                   Y_gt: batch_ys,
                                                   phase: 1,
                                                   drop_conv: 1})
                gt = np.reshape(batch_ys[0], (512, 512))
                gt = gt.astype(np.float32) * 255.
                gt = np.clip(gt, 0, 255).astype('uint8')
                cv2.imwrite("gt.bmp", gt)

                result = np.reshape(pred[0], (512, 512))
                result = result.astype(np.float32) * 255.
                result = np.clip(result, 0, 255).astype('uint8')
                cv2.imwrite("pd.bmp", result)

            if i % (DISPLAY_STEP * 10) == 0 and i:
                DISPLAY_STEP *= 10

                # train on batch
        with sess.as_default():
            _, summary = sess.run([train_op, merged_summary_op],
                                  feed_dict={X: batch_xs,
                                             Y_gt: batch_ys,
                                             lr: learning_rate,
                                             phase: 1,
                                             drop_conv: 0.8})
        summary_writer.add_summary(summary, i)
    summary_writer.close()

    save_path = saver.save(sess, "model/mutilGPUUnet2d")
    print("Model saved in file:", save_path)
    sess.close()
