from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Type, final

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetBuilder
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

__DATASET__: dict["DatasetName", Type["EvaluationDataset"]] = {}


class DatasetName(StrEnum):
    FFHQ = auto()
    IMAGENET = auto()
    CIFAR_10 = auto()
    CELEB_A = auto()


def register_dataset(name: DatasetName):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls

    return wrapper


def get_dataset(name: DatasetName, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not supported.")
    return __DATASET__[name](**kwargs)


def get_dataloader(dataset: VisionDataset, batch_size: int, num_workers: int, train: bool):
    dataloader = DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, drop_last=train
    )
    return dataloader


class EvaluationDataset(ABC):
    @property
    @abstractmethod
    def dataset_builder(self) -> DatasetBuilder | tf.data.TFRecordDataset:
        raise NotImplementedError

    @property
    @abstractmethod
    def eval_split_name(self) -> str:
        raise NotImplementedError

    @final
    def create_dataset(self, batch_size: int, shuffle_buffer_size: int = 10_000):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)

        num_epochs = 1
        prefetch_size = tf.data.experimental.AUTOTUNE
        batch_dims = [jax.local_device_count(), batch_size]

        if isinstance(self.dataset_builder, DatasetBuilder):
            self.dataset_builder.download_and_prepare()
            self.dataset_builder.download_and_prepare()
            ds: tf.data.Dataset = self.dataset_builder.as_dataset(
                split=self.eval_split_name, shuffle_files=True, read_config=read_config
            )  # type: ignore
        else:
            ds = self.dataset_builder.with_options(dataset_options)

        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(self._preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for batch_size in reversed(batch_dims):
            ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(prefetch_size)

        return ds

    def _preprocess(self, sample: dict[str, tf.Tensor]) -> dict[str, tf.Tensor | None]:
        img = self._resize_op(sample["image"])
        return dict(image=img, label=sample.get("label", None))

    def _resize_op(self, img: tf.Tensor) -> tf.Tensor:
        return img


@register_dataset(DatasetName.FFHQ)
class FFHQ(EvaluationDataset):
    image_size = (256, 256)

    @property
    def dataset_builder(self):
        return tf.data.TFRecordDataset("./assets/ffhq-r08.tfrecords")

    @property
    def eval_split_name(self):
        return "train"

    def _preprocess(self, sample: dict[str, tf.Tensor]) -> dict[str, tf.Tensor | None]:
        sample_parsed = tf.io.parse_single_example(
            sample,
            features={
                "shape": tf.io.FixedLenFeature([3], tf.int64),
                "data": tf.io.FixedLenFeature([], tf.string),
            },
        )
        data = tf.io.decode_raw(sample_parsed["data"], tf.uint8)
        data = tf.reshape(data, sample_parsed["shape"])
        data = tf.transpose(data, (1, 2, 0))
        img: tf.Tensor = tf.image.convert_image_dtype(data, tf.float32)  # type: ignore

        return dict(image=img, label=None)


@register_dataset(DatasetName.CIFAR_10)
class CIFAR10(EvaluationDataset):
    image_size = (32, 32)

    @property
    def dataset_builder(self):
        return tfds.builder("cifar10")

    @property
    def eval_split_name(self):
        return "test"

    def _resize_op(self, img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, self.image_size, antialias=True)


@register_dataset(DatasetName.IMAGENET)
class ImageNet(EvaluationDataset):
    @property
    def dataset_builder(self):
        return tfds.builder("huggingface:imagenet-1k")

    @property
    def eval_split_name(self):
        return "train"


@register_dataset(DatasetName.CELEB_A)
class CelebA(EvaluationDataset):
    image_size = (140, 140)

    @property
    def dataset_builder(self):
        return tfds.builder("celeb_a")

    @property
    def eval_split_name(self):
        return "validation"

    def _resize_op(self, img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = _central_crop(img, self.image_size[0])
        img = _resize_small(img, self.image_size[0])
        return img


def _crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
    image = tf.image.resize(
        image, size=(resolution, resolution), antialias=True, method=tf.image.ResizeMethod.BICUBIC
    )
    return tf.cast(image, tf.uint8)


def _resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def _central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)
