from torch import nn

import torch.nn.functional as F
import torch
import cv2
import numpy as np

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
from torch.nn import init
from torch.autograd import Variable
from modules.resnet import resnet34
from skimage import io, img_as_float32

def read_img(path):
    img = io.imread(path)[:,:,:3]
    img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    return img

def gaussian2kp(heatmap):
    """
    Extract the mean and from a heatmap
    """
    shape = heatmap.shape
    heatmap = heatmap.unsqueeze(-1)
    grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
    value = (heatmap * grid).sum(dim=(2, 3))
    kp = {'value': value}

    return kp

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value'] #bs*numkp*2

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()) #h*w*2
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape #1*1*h*w*2
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)  #bs*numkp*h*w*2

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out,inplace=True)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out,inplace=True)
        out = self.conv2(out)
        out += x
        return out

class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out,inplace=True)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out,inplace=True)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        del x
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        return out

class UpBlock3dN(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3dN, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        return out
class UpBlock3d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.res = ResBlock3d(out_features,kernel_size,padding)
        self.norm2 = BatchNorm3d(out_features,affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        out = self.res(out)
        out = self.norm2(out)
        out = F.relu(out,inplace=True)
        return out

class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        del x
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        out = self.pool(out)
        return out

class DownBlock3dN(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3dN, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(2, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        out = self.pool(out)
        return out

class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()

        self.res = ResBlock3d(in_features=in_features,kernel_size=kernel_size,padding=padding)
        self.norm_res = BatchNorm3d(in_features,affine=True)
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)

        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(2, 2, 2))

    def forward(self, x):
        out = self.res(x)
        out = self.norm_res(out)
        out = F.relu(out,inplace=True)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out,inplace=True)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

# class EncoderW(nn.Module):
#     """
#     Hourglass Encoder
#     """
#
#     def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
#         super(EncoderW, self).__init__()
#
#         down_blocks = []
#         for i in range(num_blocks):
#             down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
#                                            min(max_features, block_expansion * (2 ** (i + 1))),
#                                            kernel_size=3, padding=1))
#         self.down_blocks = nn.ModuleList(down_blocks)
#
#     def forward(self, x):
#         outs = [x]
#         for down_block in self.down_blocks:
#             outs.append(down_block(outs[-1]))
#         return outs

class Encoder3D(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder3D, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

class Encoder3DN(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder3DN, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3dN(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs
class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out
class DecoderW(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(DecoderW, self).__init__()

        up_blocks = []
        res_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
            if i>0:
                res_blocks.append(nn.Sequential(ResBlock2d(out_filters,kernel_size=3,padding=1),BatchNorm2d(out_filters), nn.ReLU(inplace=True)))
            else:
                res_blocks.append(nn.Sequential(ResBlock2d(in_features,kernel_size=3,padding=1),BatchNorm2d(in_features), nn.ReLU(inplace=True)))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block, res_bl in zip(self.up_blocks, self.res_blocks):
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, res_bl(skip)], dim=1)
        return out

class Decoder3DN(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder3DN, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3dN(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out

class Decoder3D(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder3D, self).__init__()

        up_blocks = []
        res_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))
            if i>0:
                res_blocks.append(nn.Sequential(ResBlock3d(out_filters,kernel_size=3,padding=1),BatchNorm3d(out_filters), nn.ReLU(inplace=True)))
            else:
                res_blocks.append(nn.Sequential(ResBlock3d(in_features,kernel_size=3,padding=1),BatchNorm3d(in_features), nn.ReLU(inplace=True)))
        self.res_blocks = nn.ModuleList(res_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block,res_bl in zip(self.up_blocks,self.res_blocks):
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, res_bl(skip)], dim=1)
        return out

class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

class HourglassW(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(HourglassW, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = DecoderW(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Hourglass3D(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass3D, self).__init__()
        self.encoder = Encoder3D(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder3D(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Hourglass3DN(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass3DN, self).__init__()
        self.encoder = Encoder3DN(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder3DN(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))
class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka


        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale))

        return out

def draw_annotation_box( image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)



class conv1d_block(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size=3,stride=2, padding=1):
        super(conv1d_block, self).__init__()
        self.conv_identity = nn.Conv1d(in_ch, out_ch, kernel_size,stride, padding)
        self.conv_gate = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = BatchNorm1d(out_ch)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self._init()
    def forward(self, x):
        identity = self.conv_identity(x)
        gate = self.sigmoid(self.conv_gate(x))
        out = identity * gate
        out = self.bn(out)
        out = self.relu(out)
        return out

    def _init(self):
        init.xavier_uniform_(self.conv_identity.weight)
        init.xavier_uniform_(self.conv_gate.weight)

class up_sample(nn.Module):
    def __init__(self, scale_factor):
        super(up_sample, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor,mode = 'linear',align_corners = True)
        return x

class transconv1d_block(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size=3,stride=1, padding=1):
        super(transconv1d_block, self).__init__()
        self.up = up_sample(2)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = BatchNorm1d(out_ch)
        self.relu = nn.LeakyReLU(inplace=True)
        self._init()
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    def _init(self):
        init.xavier_uniform_(self.conv.weight)

class GLUModel(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(GLUModel, self).__init__()
        self.glu = nn.Sequential(
            # 256→128
            conv1d_block(input_dim ,256,kernel_size=3,stride=2,padding=1),
            # 128→64
            conv1d_block(256, 256, kernel_size=3, stride=2, padding=1),
            # 64→32
            conv1d_block(256, 512, kernel_size=3, stride=2, padding=1),
            # 32→16
            conv1d_block(512, 512, kernel_size=3, stride=2, padding=1),
            # 16→32
            transconv1d_block(512,512,kernel_size=3,stride=1,padding=1),
            # 32→64
            transconv1d_block(512, 256, kernel_size=3, stride=1, padding=1),
            # 64→128
            transconv1d_block(256, 256, kernel_size=3, stride=1, padding=1),
            # 128→256
            transconv1d_block(256, 128, kernel_size=3, stride=1, padding=1),
            # 256→256
            nn.Conv1d(128,out_dim,kernel_size=3,stride=1,padding=1)
        )
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        return self.glu(x)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class MyResNet34(nn.Module):
    def __init__(self,embedding_dim,input_channel = 3):
        super(MyResNet34, self).__init__()
        self.resnet = resnet34(norm_layer = BatchNorm2d,num_classes=embedding_dim,input_channel = input_channel)
    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    model = MyResNet34(256,input_channel = 1)
    a = np.zeros([1,4,41],dtype=np.float32)

    model.cuda()
    a = torch.from_numpy(a).unsqueeze(0).cuda()
    c = model(a)
    print(c.shape)