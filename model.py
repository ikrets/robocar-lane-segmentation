import tensorflow as tf
from slim.nets.mobilenet import mobilenet_v2


def mobilenet_backbone(input_tensor, depth_multiplier,
                       output_stride, is_training, weight_decay, bn_decay):
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training, weight_decay=weight_decay)):
        with tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d],
                                       normalizer_params={'scale': True, 'center': True, 'epsilon': 1e-6,
                                                          'decay': bn_decay,
                                                          'updates_collections': None}):
            logits, endpoints = mobilenet_v2.mobilenet(input_tensor=input_tensor,
                                                       num_classes=2,
                                                       depth_multiplier=depth_multiplier,
                                                       output_stride=output_stride,
                                                       final_endpoint='layer_18')

    net = endpoints['layer_18']
    return net, endpoints


def load_mobilenet_weights(sess, checkpoint):
    to_restore = [t for t in tf.all_variables() if ('Logits' not in t.name and 'Conv_1' not in t.name)]
    saver = tf.train.Saver(var_list=to_restore, allow_empty=True)
    saver.restore(sess, checkpoint)


def segmentation_head(input_tensor, net, is_training, weight_decay, bn_decay):
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training,
                                                               weight_decay=weight_decay)):
        with tf.contrib.slim.arg_scope([tf.contrib.slim.conv2d],
                                       normalizer_params={'scale': True, 'center': True, 'epsilon': 1e-6,
                                                          'decay': bn_decay,
                                                          'updates_collections': None}):
            feature_map_size = tf.shape(net)
            branch_1 = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)
            branch_1 = tf.contrib.slim.conv2d(branch_1, 256, [1, 1], scope="image_level_conv_1x1",
                                              activation_fn=tf.nn.relu6)
            branch_1 = tf.image.resize_bilinear(branch_1, (feature_map_size[1], feature_map_size[2]),
                                                align_corners=True)

            branch_2 = tf.contrib.slim.conv2d(net, 256, [1, 1], scope='aspp0', activation_fn=tf.nn.relu6)

            out = tf.concat([branch_1, branch_2], axis=-1)
            concat_project = tf.contrib.slim.conv2d(out, 256, [1, 1], scope='concat_projection',
                                                    activation_fn=tf.nn.relu6)

            final_conv = tf.contrib.slim.conv2d(concat_project, 1, [1, 1], scope='final_layer', normalizer_fn=None,
                                                activation_fn=None,
                                                biases_initializer=tf.contrib.slim.initializers.xavier_initializer())
            out = tf.image.resize_bilinear(final_conv, (input_tensor.shape[1], input_tensor.shape[2]),
                                           align_corners=True)

            return out, {'branch_1': branch_1, 'branch_2': branch_2, 'concat_project': concat_project,
                         'final_conv': final_conv, 'resize': out}
