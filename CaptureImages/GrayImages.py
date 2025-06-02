# code provides the classes to generate Gray code (or binary) pattern images
# GrayImage  : makes the Gray code bitplane images horizontally and vertically
# BinaryImage: similar, but generates images from regular binary (non-Gray) codes

import numpy as np
from math import ceil

# currently set to, width=1920, height=1080
class GrayImage:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        max_dim = max(self.width, self.height)

        # number of bits needed for max dimension
        self.num_bits = int(ceil(np.log2(max_dim))) 

        # generate Gray code for all pixel indices
        grayCodes = np.arange(max_dim, dtype=np.uint16)
        grayCodes = (grayCodes >> 1) ^ grayCodes
        grayCodes.byteswap(inplace=True)

        # unpack bitplanes, extracat only needed bits, white = 255
        self.grayCodes = np.unpackbits(grayCodes.view(dtype=np.uint8)).reshape((-1, 16))[:, 16-self.num_bits:]*255
        self.invGrayCodes = 255 - self.grayCodes

        # out image buffer
        self.imageOut = np.empty((height, width), dtype=np.uint8)

    # generator that yeilds labeled Gray code images
    # h = horizonal, b = black, w = white, v = vertical, and i = inverted
    def getIterator(self):
        self.imageOut[:,:] = 0
        print("yielding b")
        yield "b", self.imageOut
        self.imageOut[:,:] = 255
        yield "w", self.imageOut
        for i in range(self.num_bits):
            self.imageOut[:] = self.grayCodes[:, i] # horizontal
            yield "h"+str(i), self.imageOut

            self.imageOut[:] = self.invGrayCodes[:, i]  # inverted horizontal
            yield "ih"+str(i), self.imageOut

            self.imageOut[:] = self.grayCodes[:self.height, i, None]    # vertical
            yield "v"+str(i), self.imageOut

            self.imageOut[:] = self.invGrayCodes[:self.height, i, None] # inverteed vertical
            yield "iv"+str(i), self.imageOut

    # returns a specific Gray code bitplane image       
    def getImage(self, i, inv=False, isH=True):
        if i == -1:
            self.imageOut[:] = 255
            return self.imageOut
        if not inv:
            if isH:
                self.imageOut[:] = self.grayCodes[:, i]
            else:
                self.imageOut[:] = self.grayCodes[:self.height, i, None]
        else:
            if isH:
                self.imageOut[:] = self.invGrayCodes[:, i]
            else:
                self.imageOut[:] = self.invGrayCodes[:self.height, i, None]
        return self.imageOut

# uses binary instead of Gray code, similar to GrayImage elsewise
class BinaryImage:
    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height
        max_dim = max(self.width, self.height)
        self.num_bits = int(ceil(np.log2(max_dim)))

        # binary codes, simply normal ints
        grayCodes = np.arange(max_dim, dtype=np.uint16)
        grayCodes.byteswap(inplace=True)
        self.grayCodes = np.unpackbits(grayCodes.view(dtype=np.uint8)).reshape((-1, 16))[:, 16-self.num_bits:]*255
        self.invGrayCodes = 255 - self.grayCodes

        self.imageOut = np.empty((height, width), dtype=np.uint8)

    #   uses binary bitplanes not Gray, else same as GrayImage again
    def getIterator(self):
        self.imageOut[:] = 255
        yield "w", self.imageOut
        self.imageOut[:] = 0
        yield "b", self.imageOut
        for i in range(self.num_bits):
            self.imageOut[:] = self.grayCodes[:, i]
            yield "h"+str(i), self.imageOut

            self.imageOut[:] = self.invGrayCodes[:, i]
            yield "ih"+str(i), self.imageOut

            self.imageOut[:] = self.grayCodes[:self.height, i, None]
            yield "v"+str(i), self.imageOut

            self.imageOut[:] = self.invGrayCodes[:self.height, i, None]
            yield "iv"+str(i), self.imageOut
    
    def getImage(self, i, inv=False, isH=True):
        if i == -1:
            self.imageOut[:] = 255
            return self.imageOut
        if not inv:
            if isH:
                self.imageOut[:] = self.grayCodes[:, i]
            else:
                self.imageOut[:] = self.grayCodes[:self.height, i, None]
        else:
            if isH:
                self.imageOut[:] = self.invGrayCodes[:, i]
            else:
                self.imageOut[:] = self.invGrayCodes[:self.height, i, None]
        return self.imageOut
