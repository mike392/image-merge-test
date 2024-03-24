import keras
from keras import Model, Layer
import tensorflow as tf
from keras.layers import BatchNormalization, ReLU, Conv2D, Input, Concatenate, UpSampling2D, Dropout, Add
from keras.models import Sequential

from model.config.train_config import TrainConfig


def residual_block(features=64, norm_layer=BatchNormalization):
    input = Input(shape=[TrainConfig.image_width, TrainConfig.image_height, 3])
    x = ReLU()(input)
    if norm_layer == None:
        x = Conv2D(features, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = ReLU()(x)
        x = Conv2D(features, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
    else:
        x = Conv2D(features, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = norm_layer()(x)
        x = ReLU()(x)
        x = Conv2D(features, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = norm_layer()(x)
    x = Concatenate()([x, input])
    x = ReLU()(x)
    return x

# residual_block()

# def res_unet_skip_connection_block(features, input_num=None,
#                                    submodule=None, outermost=False, innermost=False,
#                                    norm_layer=BatchNormalization, use_dropout=False, use_bias=False):
#         input = Input(shape=[TrainConfig.image_width, TrainConfig.image_height, 3])
#         if input_num is None:
#             input_num = outer_num
#         downconv = Conv2D(input_ch, inner_nc, kernel_size=3,
#                              stride=2, padding=1, bias=use_bias)


class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True,
        )
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True,
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class ResidualBlock(Layer):
    def __init__(self, name, filters, norm_layer=BatchNormalization):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.name = name
        self.conv1 = Conv2D(filters, 3, padding='same', use_bias=False, name="%s_conv1" % name)
        self.use_normalization = norm_layer is not None
        if self.use_normalization:
            self.norm1 = norm_layer()
        self.relu = ReLU()
        self.conv2 = Conv2D(filters, 3, padding='same', use_bias=False, name="%s_conv2" % name)
        if self.use_normalization:
            self.norm2 = norm_layer()

    # def build(self, input_shape):
    #     # Create weights in the build method; input_shape is automatically provided by Keras
    #     super(ResidualBlock, self).build(input_shape)  #

    def call(self, inputs):
        print("!!!Current layer - %s" % self.name)
        print("Shape 0")
        print(inputs.shape)
        x = self.relu(inputs)
        print("Shape 1")
        print(x.shape)
        x = self.conv1(x)
        print("Shape 2")
        print(x.shape)
        if self.use_normalization:
            x = self.norm1(x)
        x = self.relu(x)
        print("Shape 3")
        print(x.shape)
        x = self.conv2(x)
        if self.use_normalization:
            x = self.norm2(x)
        # x = Add()([inputs, x])
        x = self.relu(x)
        print("Shape 4")
        print(x.shape)
        return x


class ResUnetSkipConnectionBlock(Layer):
    def __init__(self, name, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=BatchNormalization, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outer_nc = outer_nc
        self.inner_nc = inner_nc
        self.name = name
        self.outermost = outermost
        self.innermost = innermost
        self.submodule = submodule
        self.use_dropout = use_dropout
        self.use_normalization = norm_layer is not None
        self.use_bias = norm_layer == InstanceNormalization

        self.downconv = Conv2D(inner_nc, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias, name="downconv_%s" % name)
        self.res_downconv = [ResidualBlock("downconv_rb1_%s" % name, inner_nc, norm_layer), ResidualBlock("downconv_rb2_%s" % name, inner_nc, norm_layer)]
        self.res_upconv = [ResidualBlock("upconv_rb1_%s" % name, outer_nc, norm_layer), ResidualBlock("upconv_rb2_%s" % name, outer_nc, norm_layer)]
        self.downrelu = ReLU()
        self.uprelu = ReLU()
        self.upsample = UpSampling2D(size=(2, 2))

        if self.use_normalization:
            self.downnorm = norm_layer()
            self.upnorm = norm_layer()

        if outermost:
            self.upconv = Conv2D(outer_nc, kernel_size=3, padding='same', use_bias=self.use_bias, name="upconv_outermost_%s" % name)
        else:
            self.upconv = Conv2D(outer_nc * 2 if not innermost else outer_nc, 3, padding='same',
                                        use_bias=self.use_bias, name="upconv_%s" % name)

        if use_dropout:
            self.dropout = Dropout(0.5)

    # def build(self, input_shape):
    #     # Create weights in the build method; input_shape is automatically provided by Keras
    #     super(ResUnetSkipConnectionBlock, self).build(input_shape)  #

    def call(self, inputs):
        x = inputs
        print("Shape before all")
        print(x.shape)
        x = self.downconv(x)
        if self.use_normalization:
            x = self.downnorm(x)
        x = self.downrelu(x)

        for block in self.res_downconv:
            x = block(x)

        if self.submodule is not None:
            print("Shape before applying submodule in %s" % self.name)
            print(x.shape)
            x = self.submodule(x)

        print("Shape before upsample")
        print(x.shape)
        x = self.upsample(x)
        x = self.upconv(x)
        print("Shape after upconv")
        print(x.shape)
        if self.use_normalization and not self.outermost:
            x = self.upnorm(x)
        x = self.uprelu(x)
        print("Shape after uprelu")
        print(x.shape)

        for block in self.res_upconv:
            x = block(x)

        if self.use_dropout:
            x = self.dropout(x)

        if self.outermost:
            return x
        else:
            return tf.concat([inputs, x], axis=-1)

def generator(output_nc, num_downs, ngf=64, norm_layer=BatchNormalization, use_dropout=False):
    x = ResUnetSkipConnectionBlock("innermost", ngf * 8, ngf * 8, submodule=None, innermost=True, norm_layer=norm_layer)
    x = ResUnetSkipConnectionBlock("middle1" % i, ngf * 8, ngf * 8, submodule=x, norm_layer=norm_layer, use_dropout=use_dropout)
    x = ResUnetSkipConnectionBlock("middle1" % i, ngf * 8, ngf * 8, submodule=x, norm_layer=norm_layer, use_dropout=use_dropout)
    x =


class ResUnetGenerator(Model):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=BatchNormalization, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        # Initialize the U-Net structure
        # The sequence of blocks will be stored in a list and then sequentially connected in the call method

        # Innermost
        unet_block = ResUnetSkipConnectionBlock("innermost", ngf * 8, ngf * 8, submodule=None, innermost=True, norm_layer=norm_layer)

        # Add middle blocks
        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock("middle%d" % i, ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        # Upsampling and expanding blocks
        sizes = [(ngf * 4, ngf * 8), (ngf * 2, ngf * 4), (ngf, ngf * 2)]
        for i, sizes in enumerate(sizes):
            unet_block = ResUnetSkipConnectionBlock("upsampling%d" % i, sizes[0], sizes[1], submodule=unet_block, norm_layer=norm_layer)

        # Outermost
        self.block = ResUnetSkipConnectionBlock("outermost", output_nc, ngf, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def call(self, inputs):
        return self.block(inputs)


generator = ResUnetGenerator(input_nc=3, output_nc=3, num_downs=7)

# generator.build(input_shape=(None, 256, 256, 3))
import numpy as np
dummy_input = np.random.random((1, 256, 256, 3))  # Batch size of 1
generator = generator(dummy_input)
generator = Model(inputs=Input(shape=dummy_input.shape), outputs=generator)
# Force building the model by passing a dummy input
print(generator.summary())
# class ResUnetGenerator(Model):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64,
#                  norm_layer=BatchNormalization, use_dropout=False):
#         super(ResUnetGenerator, self).__init__()
#         # construct unet structure
#         unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
#
#         for i in range(num_downs - 5):
#             unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#         unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
#
#         self.model = unet_block
#         self.old_lr = opt.lr
#         self.old_lr_gmm = 0.1*opt.lr
#
#     def forward(self, input):
#         return self.model(input)