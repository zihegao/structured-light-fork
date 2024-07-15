import numpy as np
from structuredlight import StructuredLight

class PhaseShifting(StructuredLight):
    def __init__(self, num=3, F=1.0):
        self.num = num
        self.F = F
        self.width = None
    
    def generate(self, dsize):
        width, height = dsize
        num = self.num
        self.width= width

        w = 2*np.pi/width * self.F

        imgs_code = (255*np.fromfunction(lambda y,x,n: 0.5*(np.cos(w*x + 2*np.pi*n/num) + 1), (height,width,num), dtype=float)).astype(np.uint8)
        
        imlist = self.split(imgs_code)
        return imlist

    def decode(self, imlist):
        img_phase = self.decodePhaseNew(imlist)
        w = 2*np.pi/self.width * self.F
        img_index = img_phase/w
        return img_index

    def decodePhase(self, imlist):
        U = self.__calcParam(imlist)
        U1, U2, U3 = self.split(U)
        img_phase = np.arccos(U2/np.sqrt(U2**2 + U3**2))
        img_phase[U3<0] = 2*np.pi - img_phase[U3<0]
        return img_phase
    
    def decodePhaseNew(self, imlist):
        U = self.__calcParam(imlist)
        U1, U2, U3 = self.split(U)

        # Compute magnitude and avoid division by zero
        magnitude = np.sqrt(U2**2 + U3**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.true_divide(U2, magnitude)
            ratio[magnitude == 0] = 0  # Handle division by zero

        # Clip ratio to avoid numerical issues with arccos
        ratio = np.clip(ratio, -1.0, 1.0)
        img_phase = np.arccos(ratio)

        # Adjust phase for negative U3 values
        img_phase[U3 < 0] = 2 * np.pi - img_phase[U3 < 0]
        return img_phase

    def decodeAmplitude(self, imlist):
        U = self.__calcParam(imlist)
        U1, U2, U3 = self.split(U)
        img_amplitude = np.sqrt(U2**2 + U3**2)
        return img_amplitude

    def decodeOffset(self, imlist):
        U = self.__calcParam(imlist)
        U1, U2, U3 = self.split(U)
        img_offset = U1
        return img_offset
    
    def __calcParam(self, imlist):
        num = len(imlist)
        
        n = np.arange(0, num)
        R = self.merge(imlist)
        M = np.array([np.ones_like(n), np.cos(2*np.pi*n/num), -np.sin(2*np.pi*n/num)]).T #(num, 3)
        M_pinv = np.linalg.inv(M.T @ M) @ M.T #(3, num)
        U = np.tensordot(M_pinv, R, axes=(1,2)).transpose(1, 2, 0) #(height, width, 3)
        return U
