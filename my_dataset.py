from collections import namedtuple
import tensorflow as tf
import augmentation

Sample = namedtuple('Sample', ('image', 'polygon'))

class MyDataset:
    def __init__(self,
                 filename,
                 batch_size,
                 shape,
                 num_readers=1,
                 num_classes=2,
                 is_training=False,
                 should_shuffle=False,
                 should_repeat=False,
                 should_augment=False):
        self.filename = filename
        self.num_classes = num_classes
        self.shape = shape
        self.batch_size = batch_size
        self.num_readers = num_readers
        self.is_training = is_training
        self.should_shuffle = should_shuffle
        self.should_repeat = should_repeat
        self.should_augment = should_augment
        self.ignore_label = 255

    def _decode_image(self, content, channels):
        return tf.cond(
            tf.image.is_jpeg(content),
            lambda: tf.image.decode_jpeg(content, channels),
            lambda: tf.image.decode_png(content, channels))

    def _parse_sample(self, sample):
        features = {
            'image':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'height':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'width':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'mask':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'name':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
        }

        parsed_features = tf.io.parse_single_example(sample, features)

        sample = {
            'image': self._decode_image(parsed_features['image'], 3),
            'image_name': parsed_features['name'],
            'height': parsed_features['height'],
            'width': parsed_features['width'],
            'label': self._decode_image(parsed_features['mask'], 1),
        }

        return sample
    
    def _preprocess_sample(self, sample):
        preprocess = lambda x:  tf.cast(x, tf.float32) / 127.0 - 1.0
        sample['image'] = preprocess(sample['image'])
        return [sample['image'], 
                tf.one_hot(tf.reshape(sample['label'], shape=(self.shape[0], self.shape[1])), 
                           depth=self.num_classes)]
    
    def _augment_sample(self, sample):
        if self.should_augment:
            sample['image'], sample['label'] = augmentation.flip_randomly_left_right_image_with_annotation(
                sample['image'],
                sample['label']
            )
            
            # sample['image'] = tf.image.random_brightness(sample['image'], 0.1)

            # sample['image'] = augmentation.distort_randomly_image_color(sample['image'])

        return sample

    def get(self):
        """Gets an iterator that iterates across the dataset once.

        Returns:
          An iterator of type tf.data.Iterator.
        """

        dataset = (
            tf.data.TFRecordDataset(self.filename)
                .map(self._parse_sample, num_parallel_calls=self.num_readers)
                .map(self._augment_sample, num_parallel_calls=self.num_readers)
                .map(self._preprocess_sample, num_parallel_calls=self.num_readers)
                )

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.should_repeat:
            dataset = dataset.repeat()  # Repeat forever for training.
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        return dataset
