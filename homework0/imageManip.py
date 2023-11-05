import math

import numpy as np
from PIL import Image
from skimage import color, io
from scipy.ndimage import affine_transform


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    # out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    ### YOUR CODE HERE
    return image[start_row:start_row + num_rows, start_col:start_col + num_cols, :]
    ### END YOUR CODE


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    # out = None

    ### YOUR CODE HERE
    out = np.clip(0.5 * image ** 2, 0, 1)
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    row_scale_factor = input_rows / output_rows
    col_scale_factor = input_cols / output_cols
    for i in range(output_rows):
        for j in range(output_cols):
            input_i = int(i * row_scale_factor)
            input_j = int(j * col_scale_factor)
            output_image[i, j, :] = input_image[input_i, input_j, :]

    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_prime = point[0] * cos_theta - point[1] * sin_theta
    y_prime = point[0] * sin_theta + point[1] * cos_theta
    rotated_point = np.array([x_prime, y_prime])
    return rotated_point
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)  # 创建与输入图像相同大小的输出图像

    # YOUR CODE HERE
    input_rows, input_cols, _ = input_image.shape

    center_x = input_cols // 2
    center_y = input_rows // 2

    # 对于输出图像的每个像素坐标 (i, j)
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            # 将像素坐标 (i, j) 转换为以中心点为原点的坐标 (x, y)
            x = j - center_x
            y = i - center_y

            # 将 (x, y) 绕原点旋转 theta 弧度得到新的坐标 (x_rot, y_rot)
            x_rot = x * np.cos(theta) - y * np.sin(theta)
            y_rot = x * np.sin(theta) + y * np.cos(theta)

            # 将旋转后的坐标 (x_rot, y_rot) 转换回以左上角为原点的坐标 (input_i, input_j)
            input_i = int(y_rot + center_y)
            input_j = int(x_rot + center_x)

            # 如果计算出的输入坐标 (input_i, input_j) 有效，则将颜色从输入图像复制到输出图像
            if 0 <= input_i < input_rows and 0 <= input_j < input_cols:
                output_image[i, j] = input_image[input_i, input_j]

    # END YOUR CODE

    # 3. Return the output image
    return output_image
