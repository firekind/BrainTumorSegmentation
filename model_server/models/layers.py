import tensorflow as tf
import tensorflow_addons as tfa
from tsalib import get_dim_vars

Layer = tf.keras.layers.Layer
GroupNormalization = tfa.layers.GroupNormalization
Activation = tf.keras.layers.Activation
Add = tf.keras.layers.Add
Conv3D = tf.keras.layers.Conv3D
UpSampling3D = tf.keras.layers.UpSampling3D
B, C, H, W, D = get_dim_vars('B C H W D')


def BlueBlock(filters: int) -> Layer:
    return Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format="channels_first",
    )


def DownSample(filters: int) -> Layer:
    return Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format="channels_first",
    )


class GreenBlock(Layer):
    def __init__(self, filters: int) -> None:
        super(GreenBlock, self).__init__()
        self.group_norm1 = GroupNormalization(
            groups=8,
            axis=1
        )
        self.relu1 = Activation('relu')
        self.conv1 = Conv3D(
            filters=filters,
            kernel_size=(3, 3, 3),
            strides=1,
            padding='same',
            data_format="channels_first",
        )

        self.group_norm2 = GroupNormalization(
            groups=8,
            axis=1
        )
        self.relu2 = Activation('relu')
        self.conv2 = Conv3D(
            filters=filters,
            kernel_size=(3, 3, 3),
            strides=1,
            padding='same',
            data_format="channels_first",
        )

        self.skip = Add()

        self.out_conv = Conv3D(
            filters=filters,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format="channels_first",
        )

    def call(self, x: (B, C, H, W, D), **kwargs):
        out = self.group_norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.group_norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return self.skip([out, self.out_conv(out)])


class UpSample(Layer):
    def __init__(self, filters: int) -> None:
        super(UpSample, self).__init__()

        self.conv = Conv3D(
            filters=filters,
            kernel_size=(1, 1, 1),
            strides=1,
            padding='same',
            data_format="channels_first",
        )

        self.up = UpSampling3D(
            size=2,
            data_format="channels_first",
        )

    def call(self, x: (B, C, H, W, D), **kwargs):
        out = self.conv(x)
        return self.up(out)
