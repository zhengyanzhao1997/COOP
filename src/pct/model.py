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


class PointTransformerCls(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, n_c, d_points):
        super().__init__()
        self.backbone = Backbone(npoints, nblocks, nneighbor, d_points)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        return res


class PointTransformerSeg(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, n_c, d_points, transformer_dim):
        super().__init__()
        self.backbone = Backbone(npoints, nblocks, nneighbor, d_points)
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


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class handsNet(nn.Module):
    def __init__(self, latent_size = 32, condition_size = 10, n_neurons = 512):
        super().__init__()
        self.enc_rb1 = ResBlock(512 + latent_size + condition_size, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons, n_neurons)
        self.enc_rb3 = ResBlock(n_neurons, n_neurons)
        self.transl = nn.Linear(n_neurons, 3)
        self.poses = nn.Linear(n_neurons, 16 * 6)
        self.dout = nn.Dropout(p=.2, inplace=False)


    def forward(self, global_feat, z, betas):
        feature = torch.cat([global_feat, z, betas],dim=-1)
        X  = self.enc_rb1(feature, True)
        X  = self.dout(X)
        X  = self.enc_rb2(X, True)
        X  = self.dout(X)
        X  = self.enc_rb3(X)
        pose = self.poses(X)
        trans = self.transl(X)
        return pose, trans


class CvaeEncoder(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, d_points, latentD, transformer_dim, condition_size):
        super().__init__()
        self.backbone = Backbone(npoints, nblocks, nneighbor, d_points, transformer_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks + condition_size, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 256))

        self.fm = nn.Linear(256, latentD)
        self.fv = nn.Linear(256, latentD)

        self.nblocks = nblocks
    
    def forward(self, x, betas):
        points, _ = self.backbone(x)
        input_f = torch.cat([points.mean(1),betas],-1)
        mean_f = self.fc1(input_f)
        means = self.fm(mean_f)
        var = self.fv(mean_f)
        return torch.distributions.normal.Normal(means, F.softplus(var))


class CvaeDecoder(nn.Module):
    def __init__(self, npoints, nblocks, nneighbor, d_points, transformer_dim):
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
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        global_feat = points.mean(1)

        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
        return self.fc3(points), global_feat



class concatCVAE(nn.Module):
    def __init__(self, latentD=16, npoints=2048, nblocks=4, nneighbor=16, n_neurons=256, use_normal=True, concat_dim=4, condition_size=10, transformer_dim=256, contact_cat=2, sigmoid=False,  **kwargs):
        super().__init__()

        d_points_enc = 3 + concat_dim
        d_points_dec = 3 + latentD
        if use_normal:
            d_points_enc += 3
            d_points_dec += 3

        self.cvae_encoder = CvaeEncoder(npoints, nblocks, nneighbor, d_points_enc, latentD, transformer_dim,)
        self.cvae_decoder = CvaeDecoder(npoints, nblocks, nneighbor, d_points_dec, transformer_dim)
        self.concat_embedding = nn.Embedding(contact_cat, concat_dim)
        self.handsnet = handsNet(latentD, condition_size, n_neurons)
        self.latentD = latentD
        self.sigmoid = sigmoid


    def forward(self, obj_points, contact_map, betas, verts_normal):
        B, N, _ = obj_points.shape
        obj_points = torch.cat([obj_points,verts_normal],dim=-1)
        
        #### encoder #####
        contact_map = self.concat_embedding(contact_map)
        obj_points_enc = torch.cat([obj_points,contact_map],dim=-1) # [B, N, 6 + concat_dim]
        P = self.cvae_encoder(obj_points_enc, betas)
        z = P.rsample()

        #### decoder #####
        obj_points_dec = torch.cat([obj_points,torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact, global_f = self.cvae_decoder(obj_points_dec)
        pose, trans = self.handsnet(global_f, z, betas)

        if self.sigmoid:
            contact = torch.sigmoid(contact)

        return {'pose':pose, 'trans':trans, 'contact_map': contact, 'mean': P.mean, 'std': P.scale}


    def inference(self, obj_points, betas, verts_normal):

        B, N, _ = obj_points.shape
        obj_points = torch.cat([obj_points,verts_normal],dim=-1)

        z_enc = torch.distributions.normal.Normal(
        loc=torch.zeros([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype),
        scale=torch.ones([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype))
        z = z_enc.rsample()
        obj_points_dec = torch.cat([obj_points,torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact, global_f = self.cvae_decoder(obj_points_dec)
        pose, trans = self.handsnet(global_f, z, betas)

        if self.sigmoid:
            contact = torch.sigmoid(contact)

        return {'pose':pose, 'trans':trans, 'contact_map': contact}


class concatCVAEH(nn.Module):
    def __init__(self, latentD=16, npoints_enc=4096, npoints_dec=2048, nblocks=4, nneighbor=16, n_neurons=256, use_normal=True, concat_dim=4, cloud_dim=4, condition_size=10, transformer_dim=256, sigmoid=False,  **kwargs):
        super().__init__()
        print('sigmoid: ',sigmoid)

        d_points_enc = 3 + concat_dim + cloud_dim
        d_points_dec = 3 + latentD  
        if use_normal:
            d_points_enc += 3
            d_points_dec += 3

        self.cvae_encoder = CvaeEncoder(npoints_enc, nblocks, nneighbor, d_points_enc, latentD, transformer_dim, condition_size)
        self.cvae_decoder = CvaeDecoder(npoints_dec, nblocks, nneighbor, d_points_dec, transformer_dim)
        self.concat_embedding = nn.Embedding(2, concat_dim)
        self.cloud_cat = nn.Embedding(2, cloud_dim)
        self.handsnet = handsNet(latentD, condition_size, n_neurons)
        self.latentD = latentD
        self.sigmoid = sigmoid
        self.obj_num = npoints_enc - 778


    def forward(self, obj_points1, verts_normal1, obj_points2, verts_normal2, hand_points, contact_map, betas, hand_normal, hand_contact):
        B, N, _ = obj_points2.shape
        
        #### encoder #####
        contact_map_obj = self.concat_embedding(contact_map)
        contact_map_hand = self.concat_embedding(hand_contact)

        obj_cloud = self.cloud_cat(torch.zeros(B,self.obj_num).long().to(obj_points1.device))
        hand_cloud = self.cloud_cat(torch.ones(B,778).long().to(obj_points1.device))

        obj_points_enc = torch.cat([obj_points1, verts_normal1, contact_map_obj, obj_cloud],dim=-1) # [B, N, 6 + concat_dim]
        hand_points_enc = torch.cat([hand_points, hand_normal,  contact_map_hand, hand_cloud],dim=-1)
        points_enc = torch.cat([obj_points_enc,hand_points_enc],dim=1)
        
        P = self.cvae_encoder(points_enc, betas)
        z = P.rsample()

        #### decoder #####
        obj_points_dec = torch.cat([obj_points2, verts_normal2],dim=-1)
        obj_points_dec = torch.cat([obj_points_dec,torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact, global_f = self.cvae_decoder(obj_points_dec)
        pose, trans = self.handsnet(global_f, z, betas)

        if self.sigmoid:
            contact = torch.sigmoid(contact)

        return {'pose':pose, 'trans':trans, 'contact_map': contact, 'mean': P.mean, 'std': P.scale}


    def inference(self, obj_points, betas, verts_normal):

        B, N, _ = obj_points.shape
        obj_points = torch.cat([obj_points,verts_normal],dim=-1)

        z_enc = torch.distributions.normal.Normal(
        loc=torch.zeros([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype),
        scale=torch.ones([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype))
        z = z_enc.rsample()
        obj_points_dec = torch.cat([obj_points,torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact, global_f = self.cvae_decoder(obj_points_dec)
        pose, trans = self.handsnet(global_f, z, betas)

        if self.sigmoid:
            contact = torch.sigmoid(contact)

        return {'pose':pose, 'trans':trans, 'contact_map': contact}



class concatCVAEH(nn.Module):
    def __init__(self, latentD=16, npoints_enc=4096, npoints_dec=2048, nblocks=4, nneighbor=16, n_neurons=256, use_normal=True, concat_dim=4, cloud_dim=4, condition_size=10, transformer_dim=256, sigmoid=False,  **kwargs):
        super().__init__()
        print('sigmoid: ',sigmoid)

        d_points_enc = 3 + concat_dim + cloud_dim
        d_points_dec = 3 + latentD
        if use_normal:
            d_points_enc += 3
            d_points_dec += 3

        self.cvae_encoder = CvaeEncoder(npoints_enc, nblocks, nneighbor, d_points_enc, latentD, transformer_dim, condition_size)
        self.cvae_decoder = CvaeDecoder(npoints_dec, nblocks, nneighbor, d_points_dec, transformer_dim)
        self.concat_embedding = nn.Embedding(2, concat_dim)
        self.cloud_cat = nn.Embedding(2, cloud_dim)
        self.handsnet = handsNet(latentD, condition_size, n_neurons)
        self.latentD = latentD
        self.sigmoid = sigmoid
        self.obj_num = npoints_enc - 778


    def forward(self, obj_points1, verts_normal1, obj_points2, verts_normal2, hand_points, contact_map, betas, hand_normal, hand_contact):
        B, N, _ = obj_points2.shape
        
        #### encoder #####
        contact_map_obj = self.concat_embedding(contact_map)
        contact_map_hand = self.concat_embedding(hand_contact)

        obj_cloud = self.cloud_cat(torch.zeros(B,self.obj_num).long().to(obj_points1.device))
        hand_cloud = self.cloud_cat(torch.ones(B,778).long().to(obj_points1.device))

        obj_points_enc = torch.cat([obj_points1, verts_normal1, contact_map_obj, obj_cloud],dim=-1) # [B, N, 6 + concat_dim]
        hand_points_enc = torch.cat([hand_points, hand_normal,  contact_map_hand, hand_cloud],dim=-1)
        points_enc = torch.cat([obj_points_enc,hand_points_enc],dim=1)
        
        P = self.cvae_encoder(points_enc,betas)
        z = P.rsample()

        #### decoder #####
        obj_points_dec = torch.cat([obj_points2, verts_normal2],dim=-1)
        obj_points_dec = torch.cat([obj_points_dec,torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact, global_f = self.cvae_decoder(obj_points_dec)
        pose, trans = self.handsnet(global_f, z, betas)

        if self.sigmoid:
            contact = torch.sigmoid(contact)

        return {'pose':pose, 'trans':trans, 'contact_map': contact, 'mean': P.mean, 'std': P.scale}


    def inference(self, obj_points, betas, verts_normal):

        B, N, _ = obj_points.shape
        obj_points = torch.cat([obj_points,verts_normal],dim=-1)

        z_enc = torch.distributions.normal.Normal(
        loc=torch.zeros([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype),
        scale=torch.ones([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype))
        z = z_enc.rsample()
        obj_points_dec = torch.cat([obj_points,torch.repeat_interleave(z.unsqueeze(1),N,1)],dim=-1)
        contact, global_f = self.cvae_decoder(obj_points_dec)
        pose, trans = self.handsnet(global_f, z, betas)

        if self.sigmoid:
            contact = torch.sigmoid(contact)

        return {'pose':pose, 'trans':trans, 'contact_map': contact}



class concatCVAEHH(nn.Module):
    def __init__(self, latentD=16, npoints_enc=4096, npoints_dec=2048, nblocks=4, nneighbor=16, n_neurons=256, use_normal=True, concat_dim=4, cloud_dim=4, condition_size=11, transformer_dim=256, sigmoid=False,  **kwargs):
        super().__init__()
        print('sigmoid: ',sigmoid)

        d_points_enc = 3 + concat_dim + cloud_dim
        d_points_dec = 3 + latentD + 1
        if use_normal:
            d_points_enc += 3
            d_points_dec += 3

        self.cvae_encoder = CvaeEncoder(npoints_enc, nblocks, nneighbor, d_points_enc, latentD, transformer_dim, condition_size)
        self.cvae_decoder = CvaeDecoder(npoints_dec, nblocks, nneighbor, d_points_dec, transformer_dim)
        self.concat_embedding = nn.Embedding(2, concat_dim)
        self.cloud_cat = nn.Embedding(2, cloud_dim)
        self.handsnet = handsNet(latentD, condition_size, n_neurons)
        self.latentD = latentD
        self.sigmoid = sigmoid
        self.obj_num = npoints_enc - 778


    def forward(self, obj_points1, verts_normal1, obj_points2, verts_normal2, hand_points, contact_map, betas, hand_normal, hand_contact, obj_height):
        B, N, _ = obj_points2.shape


        condition = torch.cat([betas,obj_height],dim=-1)
        #### encoder #####
        contact_map_obj = self.concat_embedding(contact_map)
        contact_map_hand = self.concat_embedding(hand_contact)

        obj_cloud = self.cloud_cat(torch.zeros(B,self.obj_num).long().to(obj_points1.device))
        hand_cloud = self.cloud_cat(torch.ones(B,778).long().to(obj_points1.device))

        obj_points_enc = torch.cat([obj_points1, verts_normal1, contact_map_obj, obj_cloud],dim=-1) # [B, N, 6 + concat_dim]
        hand_points_enc = torch.cat([hand_points, hand_normal,  contact_map_hand, hand_cloud],dim=-1)
        points_enc = torch.cat([obj_points_enc,hand_points_enc],dim=1)
        
        P = self.cvae_encoder(points_enc,condition)
        z = P.rsample()

        #### decoder #####
        obj_points_dec = torch.cat([obj_points2, verts_normal2],dim=-1)
        obj_points_dec = torch.cat([obj_points_dec, torch.repeat_interleave(z.unsqueeze(1),N,1), torch.repeat_interleave(obj_height.unsqueeze(1),N,1) ],dim=-1)
        contact, global_f = self.cvae_decoder(obj_points_dec)
        pose, trans = self.handsnet(global_f, z, condition)

        if self.sigmoid:
            contact = torch.sigmoid(contact)

        return {'pose':pose, 'trans':trans, 'contact_map': contact, 'mean': P.mean, 'std': P.scale}


    def inference(self, obj_points, betas, verts_normal, obj_height):

        B, N, _ = obj_points.shape
        obj_points = torch.cat([obj_points,verts_normal],dim=-1)
        condition = torch.cat([betas,obj_height],dim=-1)

        z_enc = torch.distributions.normal.Normal(
        loc=torch.zeros([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype),
        scale=torch.ones([B, self.latentD], requires_grad=False).to(obj_points.device).type(obj_points.dtype))
        z = z_enc.rsample()
        obj_points_dec = torch.cat([obj_points,torch.repeat_interleave(z.unsqueeze(1),N,1), torch.repeat_interleave(obj_height.unsqueeze(1),N,1)],dim=-1)
        contact, global_f = self.cvae_decoder(obj_points_dec)
        pose, trans = self.handsnet(global_f, z, condition)

        if self.sigmoid:
            contact = torch.sigmoid(contact)

        return {'pose':pose, 'trans':trans, 'contact_map': contact}


        

if __name__ == '__main__':
    model = concatCVAE()
    obj_points = torch.randn(4,2048,3)
    contact_map = torch.ones(4,2048).long()
    betas = torch.zeros(4,10)
    vars_network = [var[1] for var in model.named_parameters()]
    n_params = sum(p.numel() for p in vars_network if p.requires_grad)
    print('Total Trainable Parameters for network is %2.2f M.' % ((n_params) * 1e-6))