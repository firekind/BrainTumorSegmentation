from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from tsalib import get_dim_vars

from model_server import utils
from .layers import GreenBlock, BlueBlock, UpSample, DownSample

Model = tf.keras.Model
Input = tf.keras.layers.Input
SpatialDropout3D = tf.keras.layers.SpatialDropout3D
Add = tf.keras.layers.Add
Activation = tf.keras.layers.Activation
Conv3D = tf.keras.layers.Conv3D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Lambda = tf.keras.layers.Lambda
Reshape = tf.keras.layers.Reshape
GroupNormalization = tfa.layers.GroupNormalization
B, C, H, W, D = get_dim_vars("B C H W D")


class NvNet(Model):
    def __init__(self, input_shape: Tuple[int, int, int, int] = (4, 160, 192, 128), out_channels: int = 3) -> None:
        """
        Constructs the Model given in the paper "3D MRI Brain Tumor Segmentation Using AutoEncoder Regularization"
        
        Args:
            input_shape (Tuple[int, int, int, int], optional): The shape of the input. Defaults to (4, 160, 192, 128).
            out_channels (int, optional): The number of channels as output. Defaults to 3.
        """
        super(NvNet, self).__init__()

        _C, _H, _W, _D = input_shape
        assert len(input_shape) == 4, "Input shape must be a 4-tuple"
        assert (_C % 4) == 0, "The number of channels must be divisible by 4"
        assert (_H % 16) == 0 and (_W % 16) == 0 and (_D % 16) == 0, "All the input dimensions must be divisible by 16"

        self.encoder = Encoder()
        self.decoder = Decoder(out_channels)
        self.vae = VAE(input_shape)

    def call(self, x: (B, C, H, W, D), **kwargs):
        out, green1_logits, green3_logits, green5_logits = self.encoder(x)
        decoder_out = self.decoder(out, enc_green1_logits=green1_logits, enc_green3_logits=green3_logits,
                                   enc_green5_logits=green5_logits)
        vae_out, z_mean, z_var = self.vae(out)
        return decoder_out, vae_out, z_mean, z_var


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.blue1 = BlueBlock(filters=32)
        self.dropout = SpatialDropout3D(0.2)

        self.green1 = GreenBlock(filters=32)
        self.down1 = DownSample(filters=32)

        self.green2 = GreenBlock(filters=64)
        self.green3 = GreenBlock(filters=64)
        self.down2 = DownSample(filters=64)

        self.green4 = GreenBlock(filters=128)
        self.green5 = GreenBlock(filters=128)
        self.down3 = DownSample(filters=128)

        self.green6 = GreenBlock(filters=256)
        self.green7 = GreenBlock(filters=256)
        self.green8 = GreenBlock(filters=256)
        self.green9 = GreenBlock(filters=256)

    def call(self, x: (B, C, H, W, D), **kwargs):
        out = self.blue1(x)
        out = self.dropout(out)

        green1_out = self.green1(out)
        out = self.down1(green1_out)

        out = self.green2(out)
        green3_out = self.green3(out)
        out = self.down2(green3_out)

        out = self.green4(out)
        green5_out = self.green5(out)
        out = self.down3(green5_out)

        out = self.green6(out)
        out = self.green7(out)
        out = self.green8(out)
        return self.green9(out), green1_out, green3_out, green5_out


class Decoder(Model):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()

        self.up1 = UpSample(filters=128)
        self.skip1 = Add()

        self.green1 = GreenBlock(filters=128)
        self.up2 = UpSample(filters=64)
        self.skip2 = Add()

        self.green2 = GreenBlock(filters=64)
        self.up3 = UpSample(filters=32)
        self.skip3 = Add()

        self.green3 = GreenBlock(filters=32)
        self.blue1 = BlueBlock(filters=32)
        self.blue2 = BlueBlock(filters=out_channels)
        self.sigmoid = Activation('sigmoid')

    def call(self, x: (B, C, H, W, D), **kwargs):
        enc_green5_logits = kwargs["enc_green5_logits"]
        enc_green3_logits = kwargs["enc_green3_logits"]
        enc_green1_logits = kwargs["enc_green1_logits"]

        out = self.up1(x)
        out = self.skip1([out, enc_green5_logits])

        out = self.green1(out)
        out = self.up2(out)
        out = self.skip2([out, enc_green3_logits])

        out = self.green2(out)
        out = self.up3(out)
        out = self.skip3([out, enc_green1_logits])

        out = self.green3(out)
        out = self.blue1(out)
        out = self.blue2(out)
        return self.sigmoid(out)


class VAE(Model):
    def __init__(self, input_shape):
        super(VAE, self).__init__()

        _C, _H, _W, _D = input_shape

        # VD block, reduces dimensionality of data
        self.vd_group = GroupNormalization(groups=8, axis=1)
        self.vd_relu = Activation('relu')
        self.vd_conv = Conv3D(
            filters=16,
            kernel_size=(3, 3, 3),
            strides=2,
            padding='same',
            data_format='channels_first',
        )
        self.vd_flatten = Flatten()
        self.vd_dense = Dense(256)

        # VDraw, drawing values from distribution
        self.z_mean = Dense(128)
        self.z_var = Dense(128)
        self.vdraw_sample = Lambda(utils.sampling)

        # VU Block, upsizing back to a depth of 256
        self.vu_dense = Dense((_C // 4) * (_H // 16) * (_W // 16) * (_D // 16))
        self.vu_relu = Activation('relu')
        self.vu_reshape = Reshape(((_C // 4), (_H // 16), (_W // 16), (_D // 16)))
        self.vu_up1 = UpSample(filters=256)

        self.vu_up2 = UpSample(filters=128)
        self.vu_green1 = GreenBlock(filters=128)

        self.vu_up3 = UpSample(filters=64)
        self.vu_green2 = GreenBlock(filters=64)

        self.vu_up4 = UpSample(filters=32)
        self.vu_green3 = GreenBlock(filters=32)

        self.vu_blue = BlueBlock(filters=32)

        # output
        self.out_conv = Conv3D(
            filters=4,
            kernel_size=(1, 1, 1),
            strides=1,
            data_format='channels_first',
        )

    def call(self, x: (B, C, H, W, D), **kwargs):
        # VD block
        out = self.vd_group(x)
        out = self.vd_relu(out)
        out = self.vd_conv(out)
        out = self.vd_flatten(out)
        out = self.vd_dense(out)

        # Sampling portion
        mean = self.z_mean(out)
        var = self.z_var(out)
        out = self.vdraw_sample([mean, var])

        # VU block
        out = self.vu_dense(out)
        out = self.vu_relu(out)
        out = self.vu_reshape(out)
        out = self.vu_up1(out)

        out = self.vu_up2(out)
        out = self.vu_green1(out)

        out = self.vu_up3(out)
        out = self.vu_green2(out)

        out = self.vu_up4(out)
        out = self.vu_green3(out)

        out = self.vu_blue(out)

        return self.out_conv(out), mean, var
