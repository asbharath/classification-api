import tensorflow as tf
from .generic_model import GenericModel


class AI85FaceIdNet(GenericModel):
  def model(self):
    # Block 1
    self.x = self.layers().Conv2D(16//self.factor, (3, 3), padding='same', name='block1_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)

    # Block 2
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(self.x)
    self.x = self.layers().Conv2D(32//self.factor, (3, 3), padding='same', name='block2_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)

    # Block 3
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(self.x)
    self.x = self.layers().Conv2D(32//self.factor, (3, 3), padding='same', name='block3_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    
    # Block 4
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(self.x)
    self.x = self.layers().Conv2D(64//self.factor, (3, 3), padding='same', name='block4_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    
    # Block 5
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(self.x)
    self.x = self.layers().Conv2D(64//self.factor, (3, 3), padding='same', name='block5_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)
    
    # Block 6
    self.x = self.layers().Conv2D(64//self.factor, (3, 3), padding='same', name='block6_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)

    # Block 7
    self.x = self.layers().Conv2D(64//self.factor, (3, 3), padding='same', name='block7_conv1')(self.x)
    self.x = self.layers().Activation('relu')(self.x)

    # Block 8 
    self.x = self.layers().MaxPooling2D((2, 2), strides=(2, 2), name='block8_pool')(self.x)
    self.x = self.layers().Conv2D(512//self.factor, (1, 1), padding='same', name='block8_conv2')(self.x)
    self.x = self.layers().Activation('relu')(self.x)

    self.x = self.layers().AveragePooling2D((3,5))(self.x)
    self.x = self.layers().Flatten()(self.x)
