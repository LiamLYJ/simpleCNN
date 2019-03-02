import numpy as np
import conv2mul


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) // stride + 1
  out_width = (W + 2 * padding - field_width) // stride + 1
  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def conv_forward_im2col(x, w, b, conv_param):
  """
  A fast implementation of the forward pass for a convolutional layer
  based on im2col and col2im.
  """
  N, C, H, W = x.shape
  num_filters, _, filter_height, filter_width = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']

  # Check dimensions
  assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
  assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

  # Create output
  out_height = (H + 2 * pad - filter_height) // stride + 1
  out_width = (W + 2 * pad - filter_width) // stride + 1
  out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

  x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
  # x_cols = conv2mul.im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
  res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

  out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
  out = out.transpose(3, 0, 1, 2)

  cache = (x, w, b, conv_param, x_cols)
  return out, cache


img_size = 200   # Make this smaller if it runs too slow
# x = np.zeros((2, 3, img_size, img_size))
x = np.ones((2, 3, img_size, img_size))

# Set up a convolutional weights holding 2 filters, each 3x3
w = np.zeros((2, 3, 3, 3))

# The first filter converts the image to grayscale.
# Set up the red, green, and blue channels of the filter.
w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

# Second filter detects horizontal edges in the blue channel.
w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Vector of biases. We don't need any bias for the grayscale
# filter, but for the edge detection filter we want to add 128
# to each output so that nothing is negative.
b = np.array([0, 128])

out, _ = conv_forward_im2col(x, w, b, {'stride': 1, 'pad': 1})
print (out.max())
print (out.min())
