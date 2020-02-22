import numpy as np
import pywt
import matplotlib.pyplot as plt
import cv2


class Wavelet:
    def imshowgray(self, im, vmin=None, vmax=None):
        plt.imshow(im, cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)
   
    def wavMask(self, dims, scale):
        sx, sy = dims
        res = np.ones(dims)
        NM = np.round(np.log2(dims))
        for n in range(int(np.min(NM)-scale+2)//2):
            res[:int(np.round(2**(NM[0]-n))), :int(np.round(2**(NM[1]-n)))] = \
                res[:int(np.round(2**(NM[0]-n))), :int(np.round(2**(NM[1]-n)))]/2
        return res

    def imshowWAV(self, Wim, scale=1):
        plt.imshow(np.abs(Wim)*self.wavMask(Wim.shape, scale), cmap = plt.get_cmap('gray'))
   
    def coeffs2img(self, LL, coeffs):
        LH, HL, HH = coeffs
        return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))

    def unstack_coeffs(self, Wim):
            L1, L2  = np.hsplit(Wim, 2) 
            LL, HL = np.vsplit(L1, 2)
            LH, HH = np.vsplit(L2, 2)
            return LL, [LH, HL, HH]
   
    def img2coeffs(self, Wim, levels=2):
        LL, c = self.unstack_coeffs(Wim)
        coeffs = [c]
        for i in range(levels-1):
            LL, c = self.unstack_coeffs(LL)
            coeffs.insert(0,c)
        coeffs.insert(0, LL)
        return coeffs
      
    def dwt2(self, im):
        coeffs = pywt.wavedec2(im, wavelet='db4', mode='per', level=2)
        Wim, rest = coeffs[0], coeffs[1:]
        for levels in rest:
            Wim = self.coeffs2img(Wim, levels)
        return Wim

    def idwt2(self, Wim):
        coeffs = self.img2coeffs(Wim, levels=2)
        return pywt.waverec2(coeffs, wavelet='db4', mode='per')
    
    def thresholding(self, Wim, x):
        m = np.sort(abs(Wim.ravel()))[::-1]
        ndx = int(len(m)*x/100)
        thr = m[ndx]
        Wim_thr = Wim * (abs(Wim) > thr)      
        return Wim_thr

    def plotCoeff(self, Wim, Wim_thr):
        m = np.sort(abs(Wim.ravel()))[::-1]
        m_thr = np.sort(abs(Wim_thr.ravel()))[::-1]
        fig, ax = plt.subplots()
        ax.plot(m, 'b', label = 'Wavelet Coefficiants')
        ax.plot(m_thr, 'r', label = 'Thresholded Coefficiants')
        leg = ax.legend();



class Buffer:
    store = []
    
    def _init_ (self, lst = None):
        if lst == None:
            self.store = []
    
    def isFull(self, buffer_size):
        return len(self.store) == buffer_size
    
    def addToBuffer(self, wim):
        self.store.append(wim)
    
    def emptyBuffer(self):
        self.store = []
    
    def mean(self):
        return np.sum(np.array(self.store), 0)/len(self.store)


wav = Wavelet()
X = 10 #keeping only 10% of the total coefficients

'''#uncomment this section to see how wavelet thresholding works
data = cv2.cvtColor(cv2.imread('C:/Program Files/MATLAB/R2018b/toolbox/images/imdata/cameraman.tif'), cv2.COLOR_BGR2GRAY)
Wim = wav.dwt2(data)
Wim_thr = wav.thresholding(Wim, X)
im_rec = wav.idwt2(Wim_thr.reshape((256, 256)))
wav.imshowgray(data)
wav.imshowWAV(Wim)
wav.imshowWAV(Wim_thr)
wav.imshowgray(im_rec)
wav.plotCoeff(Wim, Wim_thr)
'''


###
video_path = 'C:/Program Files/MATLAB/R2018b/toolbox/vision/visiondata/visiontraffic.avi'
cv2.ocl.setUseOpenCL(False)
    
version = cv2.__version__.split('.')[0]
print(version) 

#read video file
cap = cv2.VideoCapture(video_path)
print('yes')

bg_buffer = Buffer()
bg_buffer_size = 10

## Results can be further enhanced using Morphological Processing
for _ in range(bg_buffer_size):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Wim_thr = wav.thresholding(wav.dwt2(frame), X)
    bg_buffer.addToBuffer(Wim_thr)
m, n = frame.shape[0], frame.shape[1]

bg = np.reshape(bg_buffer.mean(), (m*n, 1))
bg_buffer.emptyBuffer()
ma = np.zeros((m*n, 1))

##
# Background subtraction using Moving Average Algorithm
alpha, gamma, T = 0.001, 0.001, 128 #learning rates and a threshold to detect motion

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret == True:
        x = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        y = np.reshape(wav.thresholding(wav.dwt2(x), X), (m*n, 1))
        fore_mask = np.array(cv2.absdiff(y, bg) > T, dtype = np.uint8)
        ma = gamma*y + (1 - gamma)*ma
        bg = alpha*(y - ma) + (1 - alpha)*bg
        
        ##
        cv2.imshow('current frame', frame)
        cv2.imshow('foreground mask', wav.idwt2(fore_mask.reshape((m, n))))
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
