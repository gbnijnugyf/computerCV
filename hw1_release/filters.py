"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    #  折叠卷积核
    folded_kernel = np.flipud(np.fliplr(kernel))
    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            for m in range(-(Hk // 2), Hk // 2 + 1):
                for n in range(-(Wk // 2), Wk // 2 + 1):
                    if i + m < 0 or i + m >= Hi or j + n < 0 or j + n >= Wi:
                        continue
                    out[i, j] += image[i + m, j + n] * folded_kernel[m + Hk // 2, n + Wk // 2]
    ### END YOUR CODE

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape

    ### YOUR CODE HERE
    H, W = image.shape
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # Zero-pad the image
    padded_image = zero_pad(image, Hk // 2, Wk // 2)
    # Flip the kernel horizontally and vertically
    flipped_kernel = np.flipud(np.fliplr(kernel))
    # Compute the convolution using element-wise multiplication and np.sum()
    for i in range(Hi):
        for j in range(Wi):
            neighborhood = padded_image[i:i + Hk, j:j + Wk]
            out[i, j] = np.sum(neighborhood * flipped_kernel)
    ### END YOUR CODE

    return out


def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE
    # Compute the mean of g
    mean_g = np.mean(g)
    # Subtract the mean from g
    g_zero_mean = g - mean_g
    # Perform cross-correlation of f and g_zero_mean
    out = cross_correlation(f, g_zero_mean)
    ### END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    Hgg, Wgg = Hg // 2, Wg // 2
    out = np.zeros((Hf, Wf))
    # 填充0
    zero_pad_img = zero_pad(f,Hgg,Wgg)

    mean_g = np.mean(g)
    std_g = np.std(g)
    for i in range(Hf):
        for j in range(Wf):
            subimage = zero_pad_img[i:i + Hg, j:j + Wg]
            mean_subimage = np.mean(subimage)
            std_subimage = np.std(subimage)
            normalized_subimage = (subimage - mean_subimage) / std_subimage
            normalized_g = (g - mean_g) / std_g
            out[i, j] = np.sum(normalized_subimage * normalized_g)
    ### END YOUR CODE
    return out
