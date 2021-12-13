import numpy as np

def valid(indices, shape):
    return (indices[0] >= 0 and indices[0] < shape[0] and 
            indices[1] >= 0 and indices[1] < shape[1])


def template_func(in_img, out_img, kernel, func):
    n_rows, n_cols = in_img.shape
    condition = 255 if dilation else 0
        
    for i in range(n_rows):
        for j in range(n_cols):
            for k in kernel:
                u, v = k
                func(in_img, out_img, i, j, u, v)
    
    return out_img


def dilation(img, kernel):
    dilated_img = np.zeros(img.shape, dtype=int)
    def dilation_func(in_img, out_img, i, j, u, v):
        if valid((i+u, j+v), in_img.shape):
            out_img[i, j] = max(out_img[i, j], in_img[i+u, j+v])
    return template_func(img, dilated_img, kernel, dilation_func)


def erosion(img, kernel):
    eroded_img = np.ones(img.shape, dtype=int) * 255
    def erosion_func(in_img, out_img, i, j, u, v):
        if valid((i+u, j+v), in_img.shape):
            out_img[i, j] = min(out_img[i, j], in_img[i+u, j+v])
    return template_func(img, eroded_img, kernel, erosion_func)


def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)


def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)