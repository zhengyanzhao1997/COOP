from torch import nn
import torch
from .transformer_util import TransformerEncoder, TransformerEncoderLayer,TransformerDecoder, TransformerDecoderLayer, TransformerDecoderLayerImagen, TransformerDecoderLayerCross, TransformerRelDecoder, RelDecoderLayer
from einops import rearrange
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class condition_encoder(nn.Module):
    def __init__(self,
        hidden_size = 256,
        head_num = 8,
        num_layers = 4,
        dropout_rate = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        encoder_layer = TransformerEncoderLayer(hidden_size, head_num, dim_feedforward=  hidden_size * 4, dropout = dropout_rate, activation = 'gelu', batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_project_x = nn.Linear(10 + 1, hidden_size)
        self.linear_project_y = nn.Linear(10 + 1, hidden_size)
        self.linear_project_z = nn.Linear(10 + 1, hidden_size)
        self.linear_project = nn.Linear(hidden_size, hidden_size)
        self.cls_token = Parameter(torch.empty(1, hidden_size))

    def forward(self, wrist_transl, betas):
        bs = wrist_transl.shape[0]
        wx_input = torch.cat([wrist_transl[:,0:1], betas],dim=-1)
        wy_input = torch.cat([wrist_transl[:,1:2], betas],dim=-1)
        wz_input = torch.cat([wrist_transl[:,2:] , betas],dim=-1)
        wx_input = self.linear_project_x(wx_input).unsqueeze(1)
        wy_input = self.linear_project_y(wy_input).unsqueeze(1)
        wz_input = self.linear_project_z(wz_input).unsqueeze(1)
        
        h = torch.cat([wx_input,wy_input,wz_input,torch.repeat_interleave(self.cls_token.unsqueeze(0),bs,0)],dim=1)
        h = self.norm(h)
        h = self.encoder(h)
        return h


class condition_module(nn.Module):
    def __init__(self,
        hidden_size = 256,
        dropout_rate = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear_project_x = nn.Linear(10 + 1, hidden_size)
        self.linear_project_y = nn.Linear(10 + 1, hidden_size)
        self.linear_project_z = nn.Linear(10 + 1, hidden_size)
        self.linear_project = nn.Linear(hidden_size, hidden_size)
        self.cls_token = Parameter(torch.empty(1, hidden_size))

    def forward(self, wrist_transl, betas):
        bs = wrist_transl.shape[0]
        wx_input = torch.cat([wrist_transl[:,0:1], betas],dim=-1)
        wy_input = torch.cat([wrist_transl[:,1:2], betas],dim=-1)
        wz_input = torch.cat([wrist_transl[:,2:] , betas],dim=-1)
        wx_input = self.linear_project_x(wx_input).unsqueeze(1)
        wy_input = self.linear_project_y(wy_input).unsqueeze(1)
        wz_input = self.linear_project_z(wz_input).unsqueeze(1)
        
        encoder_token = torch.cat([wx_input,wy_input,wz_input],dim=1)
        h = self.linear_project(encoder_token)
        h = torch.cat([h,torch.repeat_interleave(self.cls_token.unsqueeze(0),bs,0)],dim=1)
        h = self.norm(h)
        return h


class motion_condition(nn.Module):
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        encoder_layers = 2,
        encoder_head_num = 2,
        flip_sin_to_cos=True,
        freq_shift=0,
        **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        decoder_layer = TransformerDecoderLayerCross(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate, activation = 'gelu', batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_project = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size),
                                            nn.GELU(),
                                            nn.Linear(2 * hidden_size, 6))
        
        self.joints_embeddings = Parameter(torch.empty(joints_num, hidden_size))
        self.condition_encoder = condition_encoder(hidden_size=hidden_size,head_num=encoder_head_num,num_layers=encoder_layers)
        self._reset_parameters()

    def forward(self, wrist_transl, wrist_orient, betas):
        bs = wrist_transl.shape[0]
        h = torch.repeat_interleave(self.joints_embeddings.unsqueeze(0),bs,0)
        h = self.norm(h)
        condition = self.condition_encoder(wrist_transl, betas)
        h = self.decoder(h, condition)
        output = self.output_project(h)
        return {'pose':output}
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class motion_condition_attention_weight(nn.Module):
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        encoder_layers = 2,
        encoder_head_num = 2,
        flip_sin_to_cos=True,
        freq_shift=0,
        **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        decoder_layer = TransformerDecoderLayerCross(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate, activation = 'gelu', batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_project = nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size),
                                            nn.GELU(),
                                            nn.Linear(2 * hidden_size, 6))
        
        self.joints_embeddings = Parameter(torch.empty(joints_num, hidden_size))
        self.condition_encoder = condition_encoder(hidden_size=hidden_size,head_num=encoder_head_num,num_layers=encoder_layers)
        self._reset_parameters()

    def forward(self, wrist_transl, wrist_orient, betas, need_weights=True):
        bs = wrist_transl.shape[0]
        h = torch.repeat_interleave(self.joints_embeddings.unsqueeze(0),bs,0)
        h = self.norm(h)
        condition = self.condition_encoder(wrist_transl, betas)
        h,w = self.decoder(h, condition, need_weights = True)
        return w
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class motion_condition_ori(nn.Module):
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        flip_sin_to_cos=True,
        freq_shift=0,
        **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        decoder_layer = TransformerDecoderLayerCross(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate, activation = 'gelu', batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_project = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Linear(4 * hidden_size, 6))
        
        self.joints_embeddings = Parameter(torch.empty(joints_num, hidden_size))
        self.condition_encoder = condition_module(hidden_size=hidden_size)
        self._reset_parameters()

    def forward(self, wrist_transl, wrist_orient, betas):
        bs = wrist_transl.shape[0]
        h = torch.repeat_interleave(self.joints_embeddings.unsqueeze(0),bs,0)
        condition = self.condition_encoder(wrist_transl, betas)
        h = self.decoder(h, condition)
        output = self.output_project(h)
        return {'pose':output}
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class motion_condition_cat(nn.Module):
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        flip_sin_to_cos=True,
        freq_shift=0,
        **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        encoder_layer = TransformerEncoderLayer(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate, activation = 'gelu', batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_project = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Linear(4 * hidden_size, 6))
        self.joints_embeddings = Parameter(torch.empty(joints_num, hidden_size))
        self.condition_project = nn.Linear(3 + 10 + 6, hidden_size)
        self._reset_parameters()

    def forward(self, wrist_transl, wrist_orient, betas):
        bs = wrist_transl.shape[0]
        h = torch.repeat_interleave(self.joints_embeddings.unsqueeze(0),bs,0)
        condition = self.condition_project(torch.cat([wrist_transl, wrist_orient, betas],dim=-1)).unsqueeze(1)
        h = torch.cat([h, condition],dim=1)
        h = self.norm(h)
        h = self.encoder(h)
        output = self.output_project(h)[:,:-1]
        return {'pose':output}
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



class condition_module_actor(nn.Module):
    def __init__(self,
        hidden_size = 256,
        dropout_rate = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear_project = nn.Sequential(nn.Linear(10 + 3, 4 * hidden_size),
                                nn.GELU(),
                                nn.Dropout(dropout_rate),
                                nn.Linear(4 * hidden_size, hidden_size))

    def forward(self, wrist_transl, betas):
        bs = wrist_transl.shape[0]
        h = torch.cat([wrist_transl, betas],dim=-1)
        h = self.linear_project(h).unsqueeze(1)
        h = self.norm(h)
        return h


class motion_condition_actor(nn.Module):
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        flip_sin_to_cos=True,
        freq_shift=0,
        **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        decoder_layer = TransformerDecoderLayer(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate, activation = 'gelu', batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_project = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Linear(4 * hidden_size, 6))
        
        self.joints_embeddings = Parameter(torch.empty(joints_num, hidden_size))
        self.condition_encoder = condition_module_actor(hidden_size=hidden_size)
        self._reset_parameters()

    def forward(self, wrist_transl, wrist_orient, betas):
        bs = wrist_transl.shape[0]
        h = torch.repeat_interleave(self.joints_embeddings.unsqueeze(0),bs,0)
        condition = self.condition_encoder(wrist_transl, betas)
        h = self.decoder(h, condition)
        output = self.output_project(h)
        return {'pose':output}
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class motion_condition_actor_rel(nn.Module):
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        flip_sin_to_cos=True,
        freq_shift=0,
        relation_map=None,
        **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        decoder_layer = RelDecoderLayer(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate)
        decoder_layer.set_relation(relation_map)

        self.decoder = TransformerRelDecoder(decoder_layer, num_layers=num_layers)
        self.output_project = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Linear(4 * hidden_size, 6))
        
        self.joints_embeddings = Parameter(torch.empty(joints_num, hidden_size))
        self.condition_encoder = condition_module_actor(hidden_size=hidden_size)
        self._reset_parameters()

    def forward(self, wrist_transl, wrist_orient, betas):
        bs = wrist_transl.shape[0]
        h = torch.repeat_interleave(self.joints_embeddings.unsqueeze(0),bs,0)
        condition = self.condition_encoder(wrist_transl, betas)
        h,w = self.decoder(h, condition)
        output = self.output_project(h)
        return {'pose':output,'attention_weight':w}
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



class pretrain_condition(nn.Module):
    def __init__(self,
        hidden_size = 256,
        dropout_rate = 0.2,
        gender_hidden = 16):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.gender_embeddings = nn.Embedding(2, gender_hidden)
        self.linear_project = nn.Sequential(nn.Linear(10 + 3 + gender_hidden, 2 * hidden_size),
                                nn.GELU(),
                                nn.Dropout(dropout_rate),
                                nn.Linear(2 * hidden_size, hidden_size))

    def forward(self, wrist_transl, gender, betas):
        gender_embedding = self.gender_embeddings(gender).squeeze(1)
        h = torch.cat([wrist_transl, betas, gender_embedding],dim=-1)
        h = self.linear_project(h).unsqueeze(1)
        h = self.norm(h)
        return h


class pretrain_actor_rel(nn.Module):
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        flip_sin_to_cos=True,
        freq_shift=0,
        relation_map=None,
        **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

        self.joints_num = joints_num

        decoder_layer = RelDecoderLayer(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate)

        decoder_layer.set_relation(relation_map)

        self.decoder = TransformerRelDecoder(decoder_layer, num_layers=num_layers)

        self.pose_embedding = nn.Sequential(nn.Linear(6, hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(hidden_size, hidden_size))

        self.output_rotation = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(4 * hidden_size, 6))

        self.output_transl = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(4 * hidden_size, 1))
        
        self.joints_embeddings = Parameter(torch.empty(joints_num + 1, hidden_size))

        self.mask_embeddings = Parameter(torch.empty(1,  hidden_size))

        self.transl_embeddings = Parameter(torch.empty(1,  hidden_size))

        self.condition_encoder = pretrain_condition(hidden_size=hidden_size)
        
        self._reset_parameters()

    def forward(self, body_pose, wrist_transl, gender, betas, mask_ids):
        
        bs = wrist_transl.shape[0]

        joints_embedding = torch.repeat_interleave(self.joints_embeddings.unsqueeze(0),bs,0) # bs, joints_num + 1 , hidden_size
        
        transl_embedding = torch.repeat_interleave(self.transl_embeddings.unsqueeze(0),bs,0) # bs,  1 , hidden_size

        mask_embedding = torch.repeat_interleave(torch.repeat_interleave(self.mask_embeddings,self.joints_num + 1, 0).unsqueeze(0),bs,0) # bs, joints_num + 1 , hidden_size

        body_embedding = self.pose_embedding(body_pose) # bs, joints_num , hidden_size

        body_embedding = torch.cat([body_embedding, transl_embedding], dim=1) # bs, joints_num + 1 , hidden_size

        mask_ids = mask_ids.unsqueeze(-1).bool() # bs, joints_num + 1 , 1
        
        body_embedding = body_embedding.where(mask_ids, mask_embedding)

        input_embedding = body_embedding + joints_embedding
        
        input_embedding = self.norm(input_embedding)

        condition = self.condition_encoder(wrist_transl, gender, betas)

        h, w = self.decoder(input_embedding, condition)

        pose_output = self.output_rotation(h[:,:-1]) # bs, joints_num, 6

        transl_output = self.output_transl(h[:,-1])

        return {'pose':pose_output, 'transl': transl_output, 'attention_weight': w}
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class pretrain_actor_norel(nn.Module):
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        flip_sin_to_cos=True,
        freq_shift=0,
        **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

        self.joints_num = joints_num

        # decoder_layer = RelDecoderLayer(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate)

        # decoder_layer.set_relation(relation_map)

        # self.decoder = TransformerRelDecoder(decoder_layer, num_layers=num_layers)

        decoder_layer = TransformerDecoderLayer(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate, activation = 'gelu', batch_first=True)
        
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.pose_embedding = nn.Sequential(nn.Linear(6, hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(hidden_size, hidden_size))

        self.output_rotation = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(4 * hidden_size, 6))

        self.output_transl = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(4 * hidden_size, 1))
        
        self.joints_embeddings = Parameter(torch.empty(joints_num + 1, hidden_size))

        self.mask_embeddings = Parameter(torch.empty(1,  hidden_size))

        self.transl_embeddings = Parameter(torch.empty(1,  hidden_size))

        self.condition_encoder = pretrain_condition(hidden_size=hidden_size)
        
        self._reset_parameters()

    def forward(self, body_pose, wrist_transl, gender, betas, mask_ids):
        
        bs = wrist_transl.shape[0]

        joints_embedding = torch.repeat_interleave(self.joints_embeddings.unsqueeze(0),bs,0) # bs, joints_num + 1 , hidden_size
        
        transl_embedding = torch.repeat_interleave(self.transl_embeddings.unsqueeze(0),bs,0) # bs,  1 , hidden_size

        mask_embedding = torch.repeat_interleave(torch.repeat_interleave(self.mask_embeddings,self.joints_num + 1, 0).unsqueeze(0),bs,0) # bs, joints_num + 1 , hidden_size

        body_embedding = self.pose_embedding(body_pose) # bs, joints_num , hidden_size

        body_embedding = torch.cat([body_embedding, transl_embedding], dim=1) # bs, joints_num + 1 , hidden_size

        mask_ids = mask_ids.unsqueeze(-1).bool() # bs, joints_num + 1 , 1
        
        body_embedding = body_embedding.where(mask_ids, mask_embedding)

        input_embedding = body_embedding + joints_embedding
        
        input_embedding = self.norm(input_embedding)

        condition = self.condition_encoder(wrist_transl, gender, betas)

        h = self.decoder(input_embedding, condition)

        pose_output = self.output_rotation(h[:,:-1]) # bs, joints_num, 6

        transl_output = self.output_transl(h[:,-1])

        return {'pose':pose_output, 'transl': transl_output}
    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class pretrain_cvae(nn.Module):
    
    def __init__(self,
        joints_num = 55,
        hidden_size = 256,
        head_num = 8,
        dropout_rate = 0.2,
        num_layers = 4,
        latentD = 32,
        flip_sin_to_cos=True,
        freq_shift=0,
        relation_map_encode=None,
        relation_map_decode=None,
        **kwargs):
        super().__init__()

        self.joints_num = joints_num

        # encoder ------------------------------------------------------------------------------------------------------
        self.norm1 = nn.LayerNorm(hidden_size)

        encoder_layer = RelDecoderLayer(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate)

        encoder_layer.set_relation(relation_map_encode)

        self.encoder = TransformerRelDecoder(encoder_layer, num_layers=num_layers)


        self.pose_embedding = nn.Sequential(nn.Linear(6, hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(hidden_size, hidden_size))

        self.transl_embedding = nn.Linear(1, hidden_size)


        self.joints_embeddings_encode = Parameter(torch.empty(joints_num + 2, hidden_size))
        

        self.z_embeddings = Parameter(torch.empty(1,  hidden_size))

        
        self.output_z = nn.Linear(hidden_size, 2 * latentD)


        self.condition_encode = pretrain_condition(hidden_size=hidden_size)


        # decoder ------------------------------------------------------------------------------------------------------

        self.norm2 = nn.LayerNorm(hidden_size)

        decoder_layer = RelDecoderLayer(hidden_size, head_num, dim_feedforward= hidden_size * 4, dropout = dropout_rate)

        decoder_layer.set_relation(relation_map_decode)

        self.decoder = TransformerRelDecoder(decoder_layer, num_layers=num_layers)

        self.z_project = nn.Linear(latentD, hidden_size)

        self.condition_decode = pretrain_condition(hidden_size=hidden_size)


        self.joints_embeddings_decode = Parameter(torch.empty(joints_num + 1, hidden_size))


        self.output_rotation = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout_rate),
                                            nn.Linear(4 * hidden_size, 6))

        self.output_transl = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                    nn.GELU(),
                                    nn.Dropout(dropout_rate),
                                    nn.Linear(4 * hidden_size, 1))


        self._reset_parameters()

    def encode(self, body_pose, transl, wrist_transl, gender, betas):

        bs = wrist_transl.shape[0]

        joints_embedding = torch.repeat_interleave(self.joints_embeddings_encode.unsqueeze(0),bs,0)

        z_embedding = torch.repeat_interleave(self.z_embeddings.unsqueeze(0),bs,0)

        body_embedding = self.pose_embedding(body_pose)

        transl_embedding = self.transl_embedding(transl).unsqueeze(1)

        input_embedding = torch.cat([body_embedding,transl_embedding,z_embedding],dim = 1)

        input_embedding += joints_embedding

        input_encoder = self.norm1(input_embedding)

        condition = self.condition_encode(wrist_transl, gender, betas)

        h_z = self.encoder(input_encoder, condition)[0][:,-1]

        mean, logvar = self.output_z(h_z).chunk(2, dim = -1)

        return torch.distributions.normal.Normal(mean, F.softplus(logvar))

    
    def decode(self, z, wrist_transl, gender, betas):

        bs = wrist_transl.shape[0]

        joints_embeddings_decode = torch.repeat_interleave(self.joints_embeddings_decode.unsqueeze(0),bs,0)

        condition = self.condition_decode(wrist_transl, gender, betas)

        add_z = torch.repeat_interleave(self.z_project(z).unsqueeze(1), self.joints_num + 1,1)

        input_decoder = self.norm2(joints_embeddings_decode + add_z)

        h, w = self.decoder(input_decoder, condition)

        pose_output = self.output_rotation(h[:,:-1]) # bs, joints_num, 6

        transl_output = self.output_transl(h[:,-1])

        return {'pose':pose_output, 'transl': transl_output, 'attention_weight': w}	

    
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


if __name__ == '__main__':
    pass