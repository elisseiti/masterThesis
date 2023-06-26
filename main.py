import os
from typing import Tuple

import numpy as np
import pyopenvdb as vdb
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.models import Model



# read all files one by one
def level_set_to_numpy(grid: vdb.FloatGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Given an input level set (in vdb format) extract the dense array representation of the volume
    and convert it to a numpy array.

    You could check the output of the numpy array by running marching cubes over the
    volume and extracting the mesh for visualization.
    """
    # Dimensions of the axis-aligned bounding box of all active voxels.
    shape = grid.evalActiveVoxelDim()
    # Return the coordinates of opposite corners of the axis-aligned bounding
    # box of all active voxels.
    start = grid.evalActiveVoxelBoundingBox()[0]
    # Create a dense array of zeros
    sdf_volume = np.zeros(shape, dtype=np.float32)
    # Copy the volume to the output, starting from the first occupied voxel
    grid.copyToArray(sdf_volume, ijk=start)
    # solve background error see OpenVDB#1096
    sdf_volume[sdf_volume < grid.evalMinMax()[0]] = grid.background

    # In order to put a mesh back into its original coordinate frame we also
    # need to know where the volume was located
    origin_xyz = grid.transform.indexToWorld(start)
    return sdf_volume, origin_xyz


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv3DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv3DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv3DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model




def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv3D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model



# Open the VDB file
folder_path = 'Dust_Devil/Dust_Devil/'
filePreFix = "embergen_dust_devil_tornado_a_"
entries = os.listdir(folder_path)
numberOfEntries = len(entries)
trainData = []

for i in range(numberOfEntries):
    path = folder_path + filePreFix + str(i) + ".vdb"
    grid = vdb.read(path, gridname='density')
    array = level_set_to_numpy(grid)
    trainData.append(array[0])


ff = [el.shape for el in trainData]
xMax = max([el[0] for el in ff])
yMax = max([el[1] for el in ff])
zMax = max([el[2] for el in ff])

print(xMax, yMax, zMax)





