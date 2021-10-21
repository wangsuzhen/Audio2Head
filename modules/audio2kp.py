from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d
from modules.util import Hourglass3D

from modules.util import gaussian2kp
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class AudioModel3D(nn.Module):
    def __init__(self,opt):
        super(AudioModel3D,self).__init__()
        self.opt = opt
        self.seq_len = opt.seq_len
        self.pad = 0

        self.down_id = AntiAliasInterpolation2d(3,0.25)
        self.down_pose = AntiAliasInterpolation2d(opt.seq_len,0.25)

        self.embedding = nn.Sequential(nn.ConvTranspose2d(1, 8, (29, 14), stride=(1, 1), padding=(0, 11)),
                                       BatchNorm2d(8),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(8, 2, (13, 13), stride=(1, 1), padding=(6, 6)))

        num_channels = 6
        self.predictor = Hourglass3D(opt.block_expansion, in_features=num_channels,
                                       max_features=opt.max_features, num_blocks=opt.num_blocks)

        self.kp = nn.Conv3d(in_channels=self.predictor.out_filters, out_channels=opt.num_kp, kernel_size=(7, 7, 7),
                            padding=(3,0,0))
        if opt.estimate_jacobian:
            self.num_jacobian_maps = opt.num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=(0,0))
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = 0.1


    def forward(self, x):
        bs,_,_,c_dim = x["audio"].shape

        audio_embedding = self.embedding(x["audio"].reshape(-1,1,4,c_dim))
        audio_embedding = F.interpolate(audio_embedding,scale_factor=2).reshape(bs,self.opt.seq_len,2,64,64).permute(0,2,1,3,4)

        id_feature = self.down_id(x["id_img"])
        pose_feature = self.down_pose(x["pose"])

        embeddings = torch.cat([audio_embedding,id_feature.unsqueeze(2).repeat(1,1,self.opt.seq_len,1,1),pose_feature.unsqueeze(1)],dim=1)

        feature_map = self.predictor(embeddings)
        feature_shape = feature_map.shape
        prediction = self.kp(feature_map).permute(0,2,1,3,4)
        prediction = prediction.reshape(-1,prediction.shape[2],prediction.shape[3],prediction.shape[4])
        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = gaussian2kp(heatmap)
        out["value"] = out["value"].reshape(-1,self.opt.seq_len,self.opt.num_kp,2)
        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map.permute(0,2,1,3,4).reshape(-1, feature_shape[1],feature_shape[3],feature_shape[4]))

            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            out["jacobian_map"] = jacobian_map
            heatmap = heatmap.unsqueeze(2)

            jacobian = heatmap * jacobian_map
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
            jacobian = jacobian.sum(dim=-1)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            out['jacobian'] = jacobian.reshape(-1,self.seq_len,self.opt.num_kp,2,2)

        out["pred_fature"] = prediction
        return out

