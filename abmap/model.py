import os
import torch
from torch import nn
import torch.nn.functional as F

try:
    from positional_encodings.torch_encodings import PositionalEncoding1D
except:
    from positional_encodings import PositionalEncoding1D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad is True)



class AbMAPAttn(nn.Module):
    def __init__(self, embed_dim = 6165, mid_dim1 = 2048, mid_dim2 = 512, mid_dim3 = 128,
                  proj_dim = 50, hidden_dim = 50, num_enc_layers = 1, num_heads=5):
        '''
            
        '''
        assert (proj_dim + 4)%num_heads == 0, "AbMAPAttn parameter proj_dim + 4 must be divisible by num_heads. i.e. (proj_dim + 4)/num_heads is an int"
        super(AbMAPAttn, self).__init__()
        self.embed_dim = embed_dim
        self.activation = nn.LeakyReLU()
        
        if embed_dim >= 2048:
            self.project = nn.Sequential(
                                nn.Linear(embed_dim, mid_dim2),
                                nn.LayerNorm(mid_dim2),
                                self.activation,
                                nn.Linear(mid_dim2, mid_dim3),
                                nn.LayerNorm(mid_dim3),
                                self.activation,
                                nn.Linear(mid_dim3, proj_dim),
                                nn.LayerNorm(proj_dim),
                                self.activation,
                                nn.Dropout(p=0.5),
                              )

        else:
            self.project = nn.Sequential(
                                nn.Linear(embed_dim, mid_dim2),
                                nn.LayerNorm(mid_dim2),
                                self.activation,
                                nn.Linear(mid_dim2, proj_dim),
                                nn.LayerNorm(proj_dim),
                                self.activation,
                                nn.Dropout(p=0.5),
                              )


        self.cossim = nn.CosineSimilarity(dim=-1)

        # last layer initialization
        self.transform2_s = nn.Linear(1, 1)
        self.transform2_s.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.transform2_s.bias = torch.nn.Parameter(torch.zeros(1))

        self.transform2_f = nn.Linear(1, 1)
        self.transform2_f.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.transform2_f.bias = torch.nn.Parameter(torch.zeros(1))


        self.enc_layer_s = nn.TransformerEncoderLayer(d_model=proj_dim+4, nhead=num_heads, batch_first=True,
                                                      dim_feedforward=proj_dim, dropout=0.5)
        self.enc_layer_f = nn.TransformerEncoderLayer(d_model=proj_dim+4, nhead=num_heads, batch_first=True,
                                                      dim_feedforward=proj_dim, dropout=0.5)

        self.attention_s = nn.TransformerEncoder(encoder_layer=self.enc_layer_s, num_layers=num_enc_layers)
        self.attention_f = nn.TransformerEncoder(encoder_layer=self.enc_layer_f, num_layers=num_enc_layers)

        self.posenc = PositionalEncoding1D(proj_dim)


    def attention(self, x, x_mask, task):
        if task == 0:
            mha = self.attention_s
        if task == 1:
            mha = self.attention_f

        if x_mask is None:
            output = mha(x)
        else:
            output = mha(x, src_key_padding_mask=x_mask)

        return output

    def mean_over_seq(self, x, x_mask):
        if x.shape[-1] > 6000:
            x = x[:,:,-2204:-4]
        else:
            x = x[:,:,:-4]

        x = torch.sum(x, dim=1)
        x = torch.div(x, torch.unsqueeze(x_mask.shape[-1] - torch.sum(x_mask, dim=-1), dim=-1))
        
        return x


    def embed(self, x, x_mask=None, task='structure', embed_type='variable'):
        assert embed_type in ['variable', 'fixed']
        assert task in ['structure', 'function']
        ### Get variablee length embedding
        # Project to a lower dimension through several linear + LayerNorm layers
        if x.shape[-1] > 6000:
            x, x_pos = x[:,:,-4-self.embed_dim:-4], x[:,:,-4:]
        else:
            x, x_pos = x[:,:,:-4], x[:,:,-4:]
        x = self.project(x)

        # ADD POSITIONAL ENCODING HERE:
        x = torch.add(x, self.posenc(x))
        x = torch.cat([x, x_pos], dim=-1)

        # Attention block
        att_task = 0 if task == 'structure' else 1
        x = self.attention(x, x_mask, att_task)

        # Output variable-length embedding as an option
        if embed_type == 'variable': return x

        ### Get Fixed-length embedding
        # method 1: mean of the features over the seq_len:
        if x_mask is None:
            x_ = torch.mean(x, dim=1)
        else:
            x_ = torch.sum(x, dim=1)
            x_ = torch.div(x_, torch.unsqueeze(x_mask.shape[-1] - torch.sum(x_mask, dim=-1), dim=-1))

        # method 2: log-sum-exp of the output features from attention layers
        x = torch.log(torch.sum(torch.exp(x), dim=1))

        # ---------------------------------------
        # concatenate mean and logsumexp
        x = F.normalize(x)
        x_ = F.normalize(x_)
        x = torch.cat((x_, x), dim=-1)
        # ---------------------------------------

        # Output fixed-length embedding here. This is the concat of the mean and LogSumExp over the residue embeddings
        if embed_type == 'fixed':
            return x

  
        
    
    def forward(self, x1, x2, x1_mask, x2_mask, task, return1 = False, return2 = False, return3 = False, 
                task_specific = False, return_cossim = False):
        
        ### Project both inputs x1 and x2 to a lower dimension through several linear + LayerNorm layers
        ### x1 and x2 are CDR-specific embeddings of antibody sequences
        if x1.shape[-1] > 6000:
            x1, x1_pos = x1[:,:,-4-self.embed_dim:-4], x1[:,:,-4:]
            x2, x2_pos = x2[:,:,-4-self.embed_dim:-4], x2[:,:,-4:]
        else:
            x1, x1_pos = x1[:,:,:-4], x1[:,:,-4:]
            x2, x2_pos = x2[:,:,:-4], x2[:,:,-4:]

        if task == 0:
            transform = self.transform2_s
        elif task == 1:
            transform = self.transform2_f

        x1 = self.project(x1)
        x2 = self.project(x2)

        if return1:
            return x1, x2

        # ADD POSITIONAL ENCODING HERE:
        x1, x2 = torch.add(x1, self.posenc(x1)), torch.add(x2, self.posenc(x2))

        # x1 = x1 - x1_pos[:,:,0:1].repeat(1,1,x1.shape[-1]) + x1_pos[:,:,2:3].repeat(1,1,x1.shape[-1])
        # x2 = x2 - x2_pos[:,:,0:1].repeat(1,1,x2.shape[-1]) + x2_pos[:,:,2:3].repeat(1,1,x2.shape[-1])

        x1 = torch.cat([x1, x1_pos], dim=-1)
        x2 = torch.cat([x2, x2_pos], dim=-1)

        # return intermediate feature 1: lower-Dim rep with positional encodings added and concatenated
        if return2:
            return x1, x2

        x1 = self.attention(x1, x1_mask, task)
        x2 = self.attention(x2, x2_mask, task)

        # return intermediate feature 2: lower-Dim rep after Transformer Encoder Block
        if return3:
            return x1, x2

        # method 1: mean of the features over the seq_len:
        if x1_mask is None:
            x1_ = torch.mean(x1, dim=1)
            x2_ = torch.mean(x2, dim=1)
        else:
            x1_ = torch.sum(x1, dim=1)
            x2_ = torch.sum(x2, dim=1)
            x1_ = torch.div(x1_, torch.unsqueeze(x1_mask.shape[-1] - torch.sum(x1_mask, dim=-1), dim=-1))
            x2_ = torch.div(x2_, torch.unsqueeze(x2_mask.shape[-1] - torch.sum(x2_mask, dim=-1), dim=-1))

        # method 2: log-sum-exp of the output features from attention layers
        x1 = torch.log(torch.sum(torch.exp(x1), dim=1))
        x2 = torch.log(torch.sum(torch.exp(x2), dim=1))

        # original normalization location
        x1, x2 = F.normalize(x1), F.normalize(x2)
        x1_, x2_ = F.normalize(x1_), F.normalize(x2_)

        # ---------------------------------------
        # concatenate mean and logsumexp
        x1 = torch.cat((x1_, x1), dim=-1)
        x2 = torch.cat((x2_, x2), dim=-1)
        # ---------------------------------------

        # x1, x2 = F.normalize(x1), F.normalize(x2)

        # Output fixed-length embedding here. These are the concat of the mean and LogSumExp over the residue embeddings
        if task_specific:
            return x1, x2

        # Take cosine similarity of two embeddings
        pred = torch.unsqueeze(self.cossim(x1, x2), dim=-1)

        if return_cossim:
            return torch.reshape(pred, (-1,)), x1, x2

        # Transform cossim for either structure or function
        pred = transform(pred)

        return torch.reshape(pred, (-1,)), x1, x2


    # def embed_dir(self, device, input_path, variable_length, task):
        # TODO - implement this?   
    #     dev = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")

    #     # Get list of files or casts single file as a list
    #     if os.path.isdir(input_path): input_iter = os.listdir(input_path)
    #     elif os.path.isfile(input_path): input_iter = list(input_path)
        
    #     outputs = []
    #     for input in input_iter:
    #         with open(input, 'rb') as p:
    #             input_embed = pickle.load(p).to(dev)
    #         input_embed = torch.unsqueeze(input_embed, 0)
    #         try:
    #             assert len(input_embed.shape) == 3
    #         except:
    #             raise ValueError("input embedding should be of shape n'(CDR length) x d")

    #         # generate the abmap embedding
    #         with torch.no_grad():
    #             if variable_length:
    #                 out_feature, _ = pretrained.embed(input_embed, task=task, embed_type='variable')
    #             else:
    #                 out_feature, _ = pretrained.embed(input_embed, task=task, embed_type='fixed')
    #         out_feature = torch.squeeze(out_feature, 0)
    #         outputs.append(out_feature)
            
    #         return outputs
            


class AbMAPLSTM(nn.Module):
    def __init__(self, embed_dim = 6165, chain_type = 'H', mid_dim1 = 2048, mid_dim2 = 512, mid_dim3 = 128,
                  proj_dim = 50, hidden_dim = 50, lstm_layers = 1):
        super(AbMAPLSTM, self).__init__()
        self.lstm_layers = lstm_layers
        self.activation = nn.LeakyReLU()
        
        self.project = nn.Sequential(
                            # nn.Linear(embed_dim, mid_dim1),
                            # nn.LayerNorm(mid_dim1),
                            # self.activation,
                            nn.Linear(embed_dim, mid_dim2),
                            nn.LayerNorm(mid_dim2),
                            self.activation,
                            nn.Linear(mid_dim2, mid_dim3),
                            nn.LayerNorm(mid_dim3),
                            self.activation,
                            nn.Linear(mid_dim3, proj_dim),
                            nn.LayerNorm(proj_dim),
                            self.activation,
                            nn.Dropout(p=0.3)
                          )

        self.rnn_struc = nn.LSTM(proj_dim + 4, hidden_dim, num_layers = lstm_layers, bidirectional=True, batch_first=True)
        self.rnn_func = nn.LSTM(proj_dim + 4, hidden_dim, num_layers = lstm_layers, bidirectional=True, batch_first=True)

        self.cossim = nn.CosineSimilarity(dim=-1)

        # last layer initialization
        self.transform2_s = nn.Linear(1, 1)
        self.transform2_s.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.transform2_s.bias = torch.nn.Parameter(torch.zeros(1))

        self.transform2_f = nn.Linear(1, 1)
        self.transform2_f.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.transform2_f.bias = torch.nn.Parameter(torch.zeros(1))


    def recurrent(self, x, task):
        if task == 0:
            rnn = self.rnn_struc
        elif task == 1:
            rnn = self.rnn_func
        output, (h_out, c_out) = rnn(x)

        # concatenate the two batch_size x hidden_dim vectors --> batch_size x (2*hidden_dim)
        h_out = h_out.view(self.lstm_layers, 2, h_out.shape[-2], h_out.shape[-1])
        out_hidden = torch.cat([h_out[-1][0], h_out[-1][1]], dim=1)

        return out_hidden


    def forward(self, x1, x2, x1_mask, x2_mask, task):
        x1, x1_pos = x1[:,:,-2204:-4], x1[:,:,-4:]
        x2, x2_pos = x2[:,:,-2204:-4], x2[:,:,-4:]

        if task == 0:
            transform = self.transform2_s
        elif task == 1:
            transform = self.transform2_f

        x1 = self.project(x1)
        x2 = self.project(x2)

        x1 = torch.cat([x1, x1_pos], dim=-1)
        x2 = torch.cat([x2, x2_pos], dim=-1)

        x1 = self.recurrent(x1, task)
        x2 = self.recurrent(x2, task)


        x1, x2 = F.normalize(x1), F.normalize(x2)

        pred = torch.unsqueeze(self.cossim(x1, x2), dim=-1)

        pred = transform(pred)

        return torch.reshape(pred, (-1,)), x1, x2


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, init_alpha, update_alpha):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.weights = nn.Parameter(torch.tensor(init_alpha), requires_grad=update_alpha) # if -2.0, alpha = e^2 = 7.xx
        self.loss_fn = nn.MSELoss()
        self.loss_s, self.loss_f = None, None
        self.alpha = None

    def forward(self, pred_s, label_s, pred_f, label_f):
        loss_s = self.loss_fn(pred_s, label_s)
        loss_f = self.loss_fn(pred_f, label_f)

        self.loss_s, self.loss_f = loss_s, loss_f

        precision_s = torch.exp(-self.weights)
        loss_s = precision_s*loss_s
        self.alpha = precision_s

        precision_f = 1/precision_s
        loss_f = precision_f*loss_f
        
        return loss_s + loss_f


class BrineyPredictor(nn.Module):
    def __init__(self, input_dim = 512, mid_dim1 = 128, mid_dim2 = 32, mid_dim3 = 8):
        super(BrineyPredictor, self).__init__()

        self.activation = nn.LeakyReLU()

        self.predictor = nn.Sequential(
                               nn.Linear(input_dim, mid_dim1),
                               self.activation,
                               nn.Linear(mid_dim1, mid_dim2),
                               self.activation,
                               nn.Linear(mid_dim2, mid_dim3),
                               self.activation,
                               # nn.Dropout(p=0.3),
                               nn.Linear(mid_dim3, 1))

    def forward(self, x):
        x = self.predictor(x)
        # x = F.relu(x)
        x = torch.exp(x)
        return torch.squeeze(x)


class BrineyPredictorAttn(nn.Module):
    def __init__(self, input_dim = 512, mid_dim1 = 128, mid_dim2 = 32, mid_dim3 = 8):
        super(BrineyPredictorAttn, self).__init__()

        self.activation = nn.LeakyReLU()

        self.predictor = nn.Sequential(
                               nn.Linear(input_dim, mid_dim1),
                               self.activation,
                               nn.Linear(mid_dim1, mid_dim2),
                               self.activation,
                               nn.Linear(mid_dim2, mid_dim3),
                               self.activation,
                               # nn.Dropout(p=0.3),
                               nn.Linear(mid_dim3, 1))

        self.enc_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True,
                                                      dim_feedforward=input_dim)

        self.trans_enc = nn.TransformerEncoder(encoder_layer=self.enc_layer, num_layers=1)

    def attention(self, x, x_mask):
        if x_mask is None:
            output = self.trans_enc(x)
        else:
            output = self.trans_enc(x, src_key_padding_mask=x_mask)

        return output


    def forward(self, x, x_mask):
        x = self.attention(x, x_mask)
        x = torch.squeeze(self.predictor(x), dim=-1)
        x = torch.logsumexp(x, dim=-1)
        x = torch.sigmoid(x)
        return x



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.01)

class DesautelsPredictor(nn.Module):
    def __init__(self, input_dim = 400, mid_dim = 64, embed_dim = 50, hidden_dim = 50, 
                 output_dim = 5, dropout= 0.3, baseline = False, num_heads=16):
        super(DesautelsPredictor, self).__init__()
        self.activation = nn.LeakyReLU()
        num_enc_layers = 1

        self.predictor = nn.Sequential(
                            nn.Linear(input_dim, mid_dim),
                            self.activation,
                            nn.Dropout(p=dropout),
                            nn.Linear(mid_dim, output_dim),
                          )

        self.enc_layer_H = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True,
                                                      dim_feedforward=hidden_dim)
        self.enc_layer_L = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True,
                                                      dim_feedforward=hidden_dim)

        self.attention_H = nn.TransformerEncoder(encoder_layer=self.enc_layer_H, num_layers=num_enc_layers)
        self.attention_L = nn.TransformerEncoder(encoder_layer=self.enc_layer_L, num_layers=num_enc_layers)

        # Xavier Uniform
        if baseline:
            self.attention_H.apply(init_weights)
            self.attention_L.apply(init_weights)


    def attention(self, x, chain_type):
        if chain_type == 'H':
            mha = self.attention_H
        else:
            mha = self.attention_L

        if x.shape[-1] > 6000:
            x = x[:,:,-2200:]

        output = mha(x)

        pooled_output = torch.mean(output, dim=1)

        return pooled_output

    def forward(self, h, wt_h, l, wt_l):

        if torch.isnan(h).any() or torch.isnan(wt_h).any() or torch.isnan(l).any() or torch.isnan(wt_l).any():
            raise ValueError

        h, wt_h = self.attention(h, 'H'), self.attention(wt_h, 'H')
        l, wt_l = self.attention(l, 'L'), self.attention(wt_l, 'L')

        x = torch.cat([h, h - wt_h, l, l - wt_l], dim=-1)

        output = self.predictor(x)
        return output
