import numpy as np

class EdgeDetection:
    def conv2d(self, img, kernel, padding=None):
        if padding is None:
            padding = kernel.shape[0] // 2
            
        expanded_img = self._add_padding(img, padding)
        result_img = np.zeros(img.shape, dtype='int32')
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                result_img[i, j] = self._convolve2d(
                        expanded_img[i: i+2*padding+1, 
                                     j: j+2*padding+1], 
                        kernel)
        
        return result_img

    
    def _add_padding(self, img, padding):
        expanded_img = np.zeros((img.shape[0] + 2*padding, 
                                 img.shape[1] + 2*padding), 
                                dtype=img.dtype)
        expanded_img[padding:-padding, padding:-padding] = img

        for i in range(padding):
            expanded_img[padding-i-1] = expanded_img[padding-i]
            expanded_img[-padding+i] = expanded_img[-padding+i-1]
            expanded_img[:, padding-i-1] = expanded_img[:, padding-i]
            expanded_img[:, -padding+i] = expanded_img[:, -padding+i-1]
            
        return expanded_img
    
    
    def _convolve2d(self, subimg, kernel):
        kernel = np.flipud(np.fliplr(kernel))
        return (subimg * kernel).sum()
    
    
    def process(self, img, threshold):
        pass