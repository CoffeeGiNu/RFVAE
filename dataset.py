import tensorflow as tf
import tensorflow_datasets as tfds


AUTOTUNE = tf.data.experimental.AUTOTUNE
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


def resize_image(x, size=(32, 32)):
    x['image'] = x['image'] / 255
    output_tensor = tf.image.resize_with_crop_or_pad(x['image'], 450, 450)
    padding = (1 - tf.image.resize_with_crop_or_pad(
        tf.ones_like(x['image']), 450, 450
        )) * tf.constant([1, 1, 1], dtype=float)
    output_tensor += padding
    output_tensor = tf.image.resize(output_tensor, (size[0], size[1]))
    output_tensor = tf.transpose(output_tensor, (2, 0, 1))
    x['image'] = output_tensor
    return x


def load_dataset_folder(path='datasets/', split='train', shuffle=True,
    batch_size=64, num_prefetch=AUTOTUNE, preprocess_fn=resize_image):
    builder = tfds.ImageFolder(path)
    dataset = builder.as_dataset(split=split, shuffle_files=shuffle)
    dataset = dataset.map(preprocess_fn).batch(batch_size)
    dataset = dataset.prefetch(num_prefetch)
    # dataset = dataset.as_numpy_iterator()
    return dataset


def cast_and_nomrmalise_images(images):
        x = images['image']
        x = (tf.cast(x, tf.float32) / 255.0)
        images['image'] = tf.transpose(x, (2, 0, 1))
        return images


def load_dataset(name='mnist', split='train', shuffle=True, slice_point=90,
    batch_size=64, num_prefetch=AUTOTUNE, preprocess_fn=None):
    if split == 'train':
        dataset = tfds.load(name=name, split=f"{split}[:{slice_point}%]")
    elif split == 'test':
        dataset = tfds.load(name=name, split=split)
    else:
        dataset = tfds.load(name=name, split=f"train[{slice_point}%:]")
    dataset = dataset.map(preprocess_fn).batch(batch_size)
    dataset = dataset.prefetch(num_prefetch).shuffle(shuffle)
    # dataset = dataset.as_numpy_iterator()
    return dataset