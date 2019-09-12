#!/bin/python3

from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import tensorflow as tf
from time import time, asctime, gmtime, localtime

from keras.applications import VGG16 as vgg16
from keras.applications.vgg16 import preprocess_input
from keras import backend as K

descr = 'Neural style transfer with Keras.'
parser = argparse.ArgumentParser(description=descr)
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025,
                    required=False, help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')
parser.add_argument('--start_noise', type=bool, default=False, required=False,
                    help='If True, start from noise, rather than content image')

args = parser.parse_args()

start_noise = args.start_noise

# Verify that TensorFlow is finding the GPU
print("\ntensorflow: {0}".format(tf.__version__), end="\n\n")
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices(), "\n")

base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight
style2content = int(style_weight/content_weight)
content2style = int(content_weight/style_weight)

# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_nrows = 256
img_ncols = int(width * img_nrows / height)

# util function to open, resize and format pictures into appropriate tensors


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# util function to convert a tensor into a valid image


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(
    preprocess_image(style_reference_image_path))

# this will contain our generated image
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

# build the VGG16 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = vgg16(input_tensor=input_tensor,
              weights='imagenet', include_top=False)
print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# The "style loss" is designed to maintain the style of the reference image in
# the generated image. It is based on the gram matrices (which capture style)
# of feature maps from the style reference image and from the generated image.


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] -
            x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] -
            x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] -
            x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] -
            x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# combine these loss functions into a single scalar
loss = K.variable(0.0)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# This Evaluator class makes it possible to compute loss and gradients in one
# pass while retrieving them via two separate functions, "loss" and "grads".
# This is done because scipy.optimize requires separate functions for loss and
# gradients, but computing them separately would be inefficient.

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# Don't print every step. Instead at integer multiples of a base b near 1.36:
exps = np.array(range(2, 22))
b = np.exp(np.log(600)/21)
print(f"b: {b:6.4f}")
printIterations = [int(b**e) for e in exps]
# printIterations.insert(0, 0)
print(printIterations)

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = preprocess_image(base_image_path)

# save initial randomized image:
img = deprocess_image(x.copy())
if style2content >= 1:
    fname = f"{result_prefix}Style2Content{style2content:03d}{0:03d}.png"
else:
    fname = f"{result_prefix}Content2Style{content2style:03d}{0:03d}.png"
save_img(fname, img)

print(asctime(localtime()))
t0 = time()
tlast = t0

# Start random number generators for numpy and TensorFlow with consistent
# values, so re-runs will generate identical output.

from numpy.random import seed
seed(3)

from tensorflow import set_random_seed
set_random_seed(4)

# Loop through multiple iterations to obtain convergence on content and style
# in generated image:

for i in range(1, iterations + 1):
    localTime = asctime(localtime())[11:19]
    if i > 1:
        print(f"{i:03d}\t{localTime}", end='')
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    ti = time()
    Δt = ti - tlast
    Δti0 = ti - t0
    if i == 1:
        print("\n  i\t       t\t\t     Δt\t\t   Δti0\t\tloss funcalls"
              "    #iters\t\t   image")
        print(f"{i:03d}\t{localTime}", end='')
    print(f"\t{int(Δt//3600):2d}h, {int((Δt % 3600)//60):2d}m,"
          f" {Δt % 60.0:4.1f}s\t{int(Δti0//3600):2d}h,"
          f" {int((Δti0 % 3600)//60):2d}m,"
          f" {Δti0 % 60.0:4.1f}s\t{min_val:012.1f}"
          f"\t    {info['funcalls']:3d}\t{info['nit']:3d}", end='')

    # save current generated image
    if i in printIterations:
        img = deprocess_image(x.copy())
        if style2content >= 1:
            fname = (f"{result_prefix}Style2Content{style2content:03d}{i:03d}"
                     ".png")
        else:
            fname = (f"{result_prefix}Content2Style{content2style:03d}{i:03d}"
                     ".png")
        print(f"\t{fname}")
        save_img(fname, img)
    else:
        print("")
    tlast = ti

print("\n+++++++++ Done! +++++++++")
