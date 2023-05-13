# -*- coding: utf-8 -*-
"""
# SplitComb class in the given code is used for splitting a 3D medical image volume into smaller sub-volumes and then combining the results back to reconstruct the original volume. This is commonly used in medical image analysis when the size of the image is too large to fit into memory or for efficient processing
The SplitComb class in the given code is used for splitting a 3D medical image volume into smaller sub-volumes and then combining the results back to reconstruct the original volume. This is commonly used in medical image analysis when the size of the image is too large to fit into memory or for efficient processing.
The split method of the class takes a 5D numpy array (batch size, depth, height, width, channels) representing a 3D medical image volume, and splits it into smaller sub-volumes with a specified side_len (default 144) and margin (default 32) using a sliding window approach. The stride used for sliding the window is specified by max_stride (default 16). The method returns a numpy array of all the sub-volumes along with the number of sub-volumes in each dimension (depth, height, width).
The combine method takes the output of the split method along with the number of sub-volumes in each dimension and reconstructs the original 3D volume by combining the sub-volumes. This is achieved by again sliding a window over the output sub-volumes with a specified stride (default 4) and margin (default 8), and combining the sub-volumes within the window to reconstruct the original volume. The reconstructed volume is returned as a numpy array of shape (depth, height, width, channels).
"""
import torch
import numpy as np


class SplitComb():
    def __init__(self, side_len, max_stride, stride, margin, pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value

    def split(self, data, side_len=None, max_stride=None, margin=None):
        if side_len == None:
            side_len = self.side_len  # 144
        if max_stride == None:
            max_stride = self.max_stride  # 16    margin=32
        if margin == None:
            margin = self.margin

        assert (side_len > margin)
        assert (side_len % max_stride == 0)
        assert (margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        nzhw = [nz, nh, nw]
        self.nzhw = nzhw

        pad = [[0, 0],
               [margin, nz * side_len - z + margin],
               [margin, nh * side_len - h + margin],
               [margin, nw * side_len - w + margin]]
        pad = np.array(pad, dtype=int)
        data = np.pad(data, pad, 'edge')  # 图像边缘值的像素填充

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, int(sz):int(
                        ez), int(sh):int(eh), int(sw):int(ew)]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, nzhw

    def combine(self, output, nzhw=None, side_len=None, stride=None, margin=None):
        if side_len == None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw
        assert (side_len % stride == 0)
        assert (margin % stride == 0)
        side_len /= stride  # 36
        margin /= stride  # 8

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones((
            int(nz * side_len),
            int(nh * side_len),
            int(nw * side_len),
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len
                    # print(splits[0].shape) # 切分后的维度(52, 52, 52, 3, 5)
                    # margin=8,side_len=36
                    split = splits[idx][int(margin):int(
                        margin + side_len), int(margin):int(margin + side_len), int(margin):int(margin + side_len)]
                    output[int(sz):int(ez), int(sh):int(
                        eh), int(sw):int(ew)] = split
                    idx += 1
        return output
