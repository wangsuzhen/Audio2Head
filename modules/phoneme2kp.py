from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d
from modules.util import Hourglass3D,Hourglass,Hourglass3DN,GLUModel,MyResNet34
from modules.keypoint_detector import KPDetector
from torch.optim.lr_scheduler import MultiStepLR
from modules.util import make_coordinate_grid,gaussian2kp



class FrameModel(nn.Module):
    def __init__(self,num_embeddings, embedding_dim):
        super(FrameModel,self).__init__()
        self.embedding_dim = embedding_dim
        if isinstance(self.embedding_dim,tuple):
            self.embedding = nn.Embedding(num_embeddings,embedding_dim[0]*embedding_dim[1])
        else:
            self.embedding = nn.Embedding(num_embeddings,embedding_dim)

class FrameModel3D(FrameModel):
    def __init__(self,opt,num_embeddings = 41, embedding_dim = (16,16)):
        super(FrameModel3D,self).__init__(num_embeddings, embedding_dim)
        self.opt = opt
        self.seq_len = opt.seq_len
        self.pad = 0

        self.down_id = AntiAliasInterpolation2d(3,0.25)
        self.down_pose = AntiAliasInterpolation2d(opt.seq_len,0.25)

        num_channels = 6
        if self.opt.nores:
            self.predictor = Hourglass3DN(opt.block_expansion, in_features=num_channels,
                                         max_features=opt.max_features, num_blocks=opt.num_blocks)
        else:
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
        phoneme_embedding = self.embedding(x["ph_w"].long())
        weight_list = torch.unsqueeze(x["w"],-1)
        phoneme_w_embedding = phoneme_embedding+phoneme_embedding*weight_list

        bs = x["ph_w"].shape[0]

        phoneme_embedding = phoneme_embedding.reshape((bs,-1,self.embedding_dim[0],self.embedding_dim[1]))
        phoneme_w_embedding = phoneme_w_embedding.reshape((bs,-1,self.embedding_dim[0],self.embedding_dim[1]))

        phoneme_embedding = F.interpolate(phoneme_embedding,scale_factor=4)#self.up(phoneme_embedding)
        phoneme_w_embedding = F.interpolate(phoneme_w_embedding,scale_factor=4)#self.up(phoneme_w_embedding)#F.interpolate(phoneme_w_embedding,scale_factor=4)
        phoneme_w_embedding = phoneme_w_embedding.unsqueeze(1)
        phoneme_embedding = phoneme_embedding.unsqueeze(1)

        embeddings = torch.cat([phoneme_embedding, phoneme_w_embedding], 1)

        # embeddings = torch.reshape(embeddings,[b])

        id_feature = self.down_id(x["id_img"])
        pose_feature = self.down_pose(x["pose"])

        embeddings = torch.cat([embeddings,id_feature.unsqueeze(2).repeat(1,1,self.opt.seq_len,1,1),pose_feature.unsqueeze(1)],dim=1)

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
