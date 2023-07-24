
import sys, os
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
from pct.pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from pct.transformer import TransformerBlock
import torch.nn.functional as F


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        

class Backbone(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, d_points, transformer_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class CvaeEncoder(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, d_points, latentD, transformer_dim):
        super().__init__()
        self.backbone = Backbone(npoints, nblocks, nneighbor, d_points, transformer_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 256))

        self.fm = nn.Linear(256, latentD)
        self.fv = nn.Linear(256, latentD)

        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        mean_f = self.fc1(points.mean(1))
        means = self.fm(mean_f)
        var = self.fv(mean_f)
        return torch.distributions.normal.Normal(means, F.softplus(var))
        


class PointTransformerSeg(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, n_c, d_points, transformer_dim):
        super().__init__()
        self.backbone = Backbone(npoints, nblocks, nneighbor, d_points, transformer_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
        return self.fc3(points)


class concatMap(nn.Module):
    def __init__(self, latentD=8, npoints=2048, nblocks=4, nneighbor=16, concat_dim=4, transformer_dim=256, n_c=6, **kwargs):
        super().__init__()

        d_points_enc = 6 + concat_dim + 1
        d_points_dec = 6 + latentD + 1

        self.cvae_encoder = CvaeEncoder(npoints, nblocks, nneighbor, d_points_enc, latentD, transformer_dim)
        self.cvae_decoder = PointTransformerSeg(npoints, nblocks, nneighbor, n_c, d_points_dec, transformer_dim)
        self.concat_embedding = nn.Embedding(n_c, concat_dim)
        self.latentD = latentD


    def forward(self, points, normal, contact_map, obj_height):
        B, N, _ = points.shape

        #### encoder #####
        obj_height = torch.repeat_interleave(obj_height.unsqueeze(1),N,1)
        contact_map_obj = self.concat_embedding(contact_map)

        obj_points = torch.cat([points, normal, obj_height],dim=-1) # [B, N, 6 + concat_dim]
        points_enc = torch.cat([obj_points, contact_map_obj],dim=-1)
        P = self.cvae_encoder(points_enc)
        z = P.rsample()

        #### decoder #####
        points_dec = torch.cat([obj_points, torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact = self.cvae_decoder(points_dec)
        return {'contact_map': contact, 'mean': P.mean, 'std': P.scale}


    def infer(self, points, normal, obj_height):
        B, N, _ = points.shape
        obj_height = torch.repeat_interleave(obj_height.unsqueeze(1),N,1)
        obj_points = torch.cat([points, normal, obj_height],dim=-1) # [B, N, 6 + concat_dim]
        z_enc = torch.distributions.normal.Normal(
        loc=torch.zeros([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype),
        scale=torch.ones([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype))
        z = z_enc.rsample()
        points_dec = torch.cat([obj_points, torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact = self.cvae_decoder(points_dec)
        return {'contact_map': contact}



class concatMapAction(nn.Module):
    def __init__(self, latentD=8, npoints=2048, nblocks=4, nneighbor=16, concat_dim=4, transformer_dim=256, n_c=6, action_cat=2, action_dim=4, **kwargs):
        super().__init__()

        d_points_enc = 6 + concat_dim + action_dim
        d_points_dec = 6 + latentD + action_dim

        self.cvae_encoder = CvaeEncoder(npoints, nblocks, nneighbor, d_points_enc, latentD, transformer_dim)
        self.cvae_decoder = PointTransformerSeg(npoints, nblocks, nneighbor, n_c, d_points_dec, transformer_dim)
        self.concat_embedding = nn.Embedding(n_c, concat_dim)
        self.action_embedding = nn.Embedding(action_cat, action_dim)
        self.latentD = latentD


    def forward(self, points, normal, contact_map, action):
        B, N, _ = points.shape

        #### encoder #####
        contact_map_obj = self.concat_embedding(contact_map)
        action_ebd = self.action_embedding(action)
        action_ebd =  torch.repeat_interleave(action_ebd.unsqueeze(1),N,1)
        obj_points = torch.cat([points, normal, action_ebd],dim=-1) # [B, N, 6 + concat_dim]

        points_enc = torch.cat([obj_points, contact_map_obj],dim=-1)
        P = self.cvae_encoder(points_enc)
        z = P.rsample()

        #### decoder #####
        points_dec = torch.cat([obj_points, torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact = self.cvae_decoder(points_dec)
        return {'contact_map': contact, 'mean': P.mean, 'std': P.scale}


    def infer(self, points, normal, action, z = None):
        B, N, _ = points.shape
        action_ebd = self.action_embedding(action)
        action_ebd =  torch.repeat_interleave(action_ebd.unsqueeze(1),N,1)
        obj_points = torch.cat([points, normal, action_ebd],dim=-1) # [B, N, 6 + concat_dim]
        
        if z is None:
            z_enc = torch.distributions.normal.Normal(
            loc=torch.zeros([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype),
            scale=torch.ones([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype))
            z = z_enc.rsample()

        points_dec = torch.cat([obj_points, torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact = self.cvae_decoder(points_dec)
        return {'contact_map': contact}


class Point2Hand(nn.Module):
    def __init__(self, npoints=4096, nblocks=4, nneighbor=16, concat_dim=4, n_c=6, condition_size=10, transformer_dim=256, **kwargs):
        super().__init__()

        d_points = 6 + concat_dim + 1
        self.backbone = Backbone(npoints, nblocks, nneighbor, d_points, transformer_dim)
        self.fc = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks + condition_size, 512),
            nn.ReLU(),
            nn.Dropout(p=.2, inplace=False),
            nn.Linear(512, 256),
        )
        self.concat_embedding = nn.Embedding(n_c, concat_dim)
        self.poses = nn.Linear(256, 16*6)
        self.trans = nn.Linear(256, 3)
        self.nblocks = nblocks
    
    def forward(self, points, normal, contact_map, betas, obj_height):
        B, N, _ = points.shape
        obj_height = torch.repeat_interleave(obj_height.unsqueeze(1),N,1)
        contact_map_obj = self.concat_embedding(contact_map)
        obj_points_enc = torch.cat([points, normal, contact_map_obj, obj_height],dim=-1)
        points, _ = self.backbone(obj_points_enc)
        global_feat = torch.cat([points.mean(1), betas],dim=-1)
        global_feat = self.fc(global_feat)
        pose = self.poses(global_feat)
        trans = self.trans(global_feat)
        return {'pose':pose, 'trans':trans}


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))