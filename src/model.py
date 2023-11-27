import torch
from torch import nn


class Transperf(torch.nn.Module):
    def __init__(self, 
                 input_size=13, 
                 nhead=8,
                 d_model=16,
                 dim_feedforward=32,
                 dropout=0.1,
                 num_encoder_layers=4,
                 load_model=None,):
        super(Transperf, self).__init__()
        if input_size % nhead != 0:
            input_size = (input_size // nhead)*nhead + nhead
        print("input size: {}, nhead: {}, d_model: {}".format(input_size, nhead, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        
        self.embedding = nn.Linear(2, d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.mlp = nn.Linear(d_model*input_size, 1)
        self.d_model = d_model
        self.nhead = nhead
        if load_model is not None:
            self.load_model(load_model)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.mlp(output.flatten(start_dim=1))
        return output

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        
        
