import torch
import torch.nn as nn


class InputAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_e = nn.Linear(hidden_dim*2 + 1, 64)
        self.v_e = nn.Linear(64, 1)
        self.input_dim = input_dim

    def forward(self, x, h_prev, s_prev):
        # x: [batch, input_dim]
        # h_prev, s_prev: [batch, hidden_dim]
        scores = []
        for k in range(self.input_dim):
            feat = x[:, k].unsqueeze(1)  # [batch, 1]
            cat = torch.cat([h_prev, s_prev, feat], dim=1)
            score = self.v_e(torch.tanh(self.W_e(cat)))  # [batch, 1]
            scores.append(score)
        scores = torch.cat(scores, dim=1)  # [batch, input_dim]
        alpha = torch.softmax(scores, dim=1)
        x_tilde = (alpha * x)  # [batch, input_dim]
        return x_tilde, alpha



class TemporalAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.W_d = nn.Linear(dec_hidden_dim*2 + enc_hidden_dim, 64)
        self.v_d = nn.Linear(64, 1)

    def forward(self, encoder_hiddens, d_prev, c_prev):
        # encoder_hiddens: [batch, lookback, enc_hidden_dim]
        batch, seq_len, _ = encoder_hiddens.size()
        d_prev = d_prev.unsqueeze(1).repeat(1, seq_len, 1)
        c_prev = c_prev.unsqueeze(1).repeat(1, seq_len, 1)
        attn_in = torch.cat([encoder_hiddens, d_prev, c_prev], dim=-1)
        energy = torch.tanh(self.W_d(attn_in))
        score = self.v_d(energy).squeeze(-1)
        beta = torch.softmax(score, dim=1)
        context = torch.bmm(beta.unsqueeze(1), encoder_hiddens).squeeze(1)
        return context, beta

class DA_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, horizon=1, lookback=10, encoder_layers=1, decoder_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.lookback = lookback
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # Stacking LSTMCells manually if more than 1 layer:
        self.encoder_cells = nn.ModuleList([nn.LSTMCell(input_dim if i==0 else hidden_dim, hidden_dim) for i in range(encoder_layers)])
        self.input_attn = InputAttention(input_dim, hidden_dim)
        self.decoder_cells = nn.ModuleList([nn.LSTMCell(1 if i==0 else hidden_dim, hidden_dim) for i in range(decoder_layers)])
        self.temporal_attn = TemporalAttention(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + hidden_dim + lookback, output_dim)

    def forward(self, X, y_hist):
        batch_size = X.size(0)
        device = X.device

        # ENCODER WITH INPUT ATTENTION & STACKED LSTMCell
        h_enc = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.encoder_layers)]
        c_enc = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.encoder_layers)]
        enc_hiddens = []
        for t in range(self.lookback):
            x_t = X[:, t, :]  # [batch, input_dim]
            x_tilde, alpha = self.input_attn(x_t, h_enc[-1], c_enc[-1])
            h_new, c_new = [], []
            h, c = x_tilde, None
            for i, cell in enumerate(self.encoder_cells):
                h, c = cell(h, (h_enc[i], c_enc[i]))
                h_new.append(h)
                c_new.append(c)
            h_enc, c_enc = h_new, c_new
            enc_hiddens.append(h_enc[-1].unsqueeze(1))  # Only last layer output
        enc_hiddens = torch.cat(enc_hiddens, dim=1)  # [batch, lookback, hidden_dim]

        # DECODER WITH TEMPORAL ATTENTION & STACKED LSTMCell
        d_dec = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.decoder_layers)]
        c_dec = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.decoder_layers)]
        y_t_prev = y_hist[:, -1].unsqueeze(1)  # [batch, 1]
        outputs = []
        for t in range(self.horizon):
            # Attention over encoder hidden states
            context, beta = self.temporal_attn(enc_hiddens, d_dec[-1], c_dec[-1])  # use top decoder layer
            # Stacked decoder LSTMCell(s)
            h, c = y_t_prev, None
            h_new, c_new = [], []
            for i, cell in enumerate(self.decoder_cells):
                h, c = cell(h, (d_dec[i], c_dec[i]))
                h_new.append(h)
                c_new.append(c)
            d_dec, c_dec = h_new, c_new
            # FC input: decoder state, context, and input history
            fc_input = torch.cat([d_dec[-1], context, y_hist], dim=1)
            out = self.fc(fc_input)
            outputs.append(out.unsqueeze(1))
            y_t_prev = out  # Autoregressive
        outputs = torch.cat(outputs, dim=1)  # [batch, horizon, output_dim]
        return outputs
