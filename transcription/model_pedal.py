import torch as th
from torch import nn
import numpy as np
from torch.nn import functional as F
import nnAudio
from torchaudio import transforms

# from .cqt import CQT
from .constants import SR, HOP
from .context import random_modification, update_context
# from .cqt import MultiCQT
from .midispectrogram import CombinedSpec, MidiSpec

class PedalModel(nn.Module):
    def __init__(self, config, perceptual_w=False):
        super().__init__()
        self.model = config.model
        self.win_fw = config.win_fw
        self.win_bw = config.win_bw
        self.n_fft = config.n_fft
        self.hidden_per_pitch = config.hidden_per_pitch
        self.context_len = self.win_fw + self.win_bw + 1
        self.pitchwise = config.pitchwise_lstm
        if 'CQT' in self.model:
            self.frontend = MultiCQT()
        elif 'Mel2' in self.model:
            self.frontend = CombinedSpec()
        elif 'MIDI' in self.model:
            self.frontend = MIDIFrontEnd(config.n_per_pitch)
        else:
            self.frontend = transforms.MelSpectrogram(sample_rate=SR, n_fft=config.n_fft,
                hop_length=HOP, f_min=config.f_min, f_max=config.f_max, n_mels=config.n_mels, normalized=False)
        if 'CNN' == self.model:
            self.acoustic = CNN(config.n_mels, config.cnn_unit, config.fc_unit, config.hidden_per_pitch) 
        else:
            raise KeyError(f'wrong model:{self.model}')
            
        self.lstm = nn.LSTM(config.fc_unit, config.lstm_unit, num_layers=2, batch_first=False, bidirectional=True)
        self.output = nn.Linear(config.lstm_unit*2, 5)
            

    def forward(self, audio):
        max_step = 400
        audio_len = audio.shape[1]
        batch_size = audio.shape[0]
        step_len = (audio_len - 1) // HOP+ 1
        device = audio.device

        if step_len > max_step:
            frame = th.zeros((batch_size, step_len, 5)).to(device)
            n_segs = ((step_len - 1)//max_step + 1)
            if 0 <= audio_len - (n_segs-1)*max_step* HOP< self.n_fft//2: # padding size of cqt
                n_segs -= 1
            seg_edges = [el*max_step for el in range(n_segs)]
            step_len = audio.shape[1] // HOP
            if step_len > 400:
                seg_num = 0
                for n, step in enumerate(seg_edges[:-1]):
                    seg_num += 1
                    offset = step
                    if step == 0:  # First segment
                        unpad_start = False
                        start = 0
                    else:
                        del conv_out
                        unpad_start = True
                        start = offset * HOP - self.n_fft//2 

                    if step == seg_edges[-1]:  # Last segment
                        unpad_end = False
                        end = None
                        seg_end = step_len
                    else:
                        # margin for CNN
                        end = (offset + max_step + 10) * HOP + self.n_fft//2
                        unpad_end = True
                        seg_end = seg_edges[n+1]
                    
                    conv_out = self.local_forward(
                        audio[:, start: end],
                        unpad_start=unpad_start, unpad_end=unpad_end)
                    conv_out = conv_out.permute(1, 0, 2) # T x B x hidden
                    lstm_out, lstm_hidden = self.lstm(conv_out) # T x B x hidden*2
                    frame_out = self.output(lstm_out) # T, B x n_class
                    frame_out = frame_out.permute(1, 0, 2) # B x T x n_class
                    frame[:, offset:seg_end] = frame_out[:, :seg_end-offset]
            return frame
        else:             
            conv_out = self.local_forward(audio)  # B x T x hidden
            conv_out = conv_out.permute(1, 0, 2) # T x B x hidden
            lstm_out, lstm_hidden = self.lstm(conv_out) # T x B x hidden*2
            # print(f'lstm:{lstm_out.shape}')
            frame_out = self.output(lstm_out) # T, B x n_class
            frame_out = frame_out.permute(1, 0, 2) # B x T x n_class
            return frame_out

    def local_forward(self, audio, unpad_start=False, unpad_end=False):
        if 'MIDI' in self.model:
            spec = self.frontend(audio)
            if unpad_start:
                spec = spec[:,:,:,self.n_fft//2//HOP:]
            if unpad_end:
                spec = spec[:,:,:,:-self.n_fft//2//HOP]
            conv_out = self.acoustic(spec)  # B x T x hidden_per_pitch x 88
            return conv_out
        else:
            mel = self.frontend(
                audio[:, :-1]).transpose(-1, -2) # B L F
            mel = (th.log(th.clamp(mel, min=1e-9)) + 7) / 7
            if unpad_start:
                mel = mel[:,self.n_fft//2//HOP:]
            if unpad_end:
                mel = mel[:,:-self.n_fft//2//HOP]
            conv_out = self.acoustic(mel)  # B x T x hidden_per_pitch x 88

            return conv_out

class FilmLayer(nn.Module):
    def __init__(self, n_f, channel, hidden=16):
        super().__init__()
        pitch = (th.arange(n_f)/n_f).view(n_f, 1)
        self.register_buffer('pitch', pitch.float())
        self.alpha_linear = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channel, bias=False),
        )
        self.beta_linear = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channel, bias=False),
        )

    def forward(self, x):
        # x : shape of (B,C,L,F)
        alpha = self.alpha_linear(self.pitch).transpose(0,1).unsqueeze(1) # C, 1, F
        beta = self.beta_linear(self.pitch).transpose(0,1).unsqueeze(1) # C, 1, F
        x = alpha * x + beta
        return x


class FilmBlock(nn.Module):
    def __init__(self, n_input, n_unit, n_f, hidden=16, use_film=True, width_l1=3, width_l2=3):
        super().__init__()
        assert(width_l1 in [1,3])
        assert(width_l2 in [1,3])
        self.conv1 = nn.Conv2d(n_input, n_unit, (3, width_l1), padding='same')
        self.conv2 = nn.Conv2d(n_unit, n_unit, (3, width_l2), padding='same')
        self.bn = nn.BatchNorm2d(n_unit)
        self.use_film = use_film
        if use_film:
            self.film = FilmLayer(n_f, n_unit, hidden=hidden)

    def forward(self, x):
        if self.use_film:
            # x : shape of B C F L
            x = F.relu(self.conv1(x))
            x = self.film(x.transpose(2,3)).transpose(2,3)
            res = self.conv2(x)
            res = self.bn(res) 
            res = self.film(res.transpose(2,3)).transpose(2,3)
            x = F.relu(x + res)
            return x
        else:
            x = F.relu(self.conv1(x))
            res = self.conv2(x)
            res = self.bn(res) 
            x = F.relu(x + res)
            return x

class CNN(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit, hidden_per_pitch):
        super().__init__()
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=True),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=True),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.layernorm = nn.LayerNorm(fc_unit)

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C

        x = self.layernorm(fc_x)
        return F.relu(x)

class PAR(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch*88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = fc_x.transpose(1,2)  # B C L
        fc_x = F.pad(fc_x, (self.win_bw, self.win_fw)).unsqueeze(3)
        fc_x = F.unfold(fc_x, (self.win_bw + self.win_fw + 1, 1)) 
        fc_x = self.win_fc(fc_x.transpose(1,2))  # B L C
        fc_x = fc_x.view(batch_size, -1, self.hidden_per_pitch, 88)

        x = self.layernorm(fc_x)
        return F.relu(x)
    
class PAR_v2(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film,
                 cnn_widths = [3,3,3,3,3,3], multifc=True):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[0], width_l2=cnn_widths[1]),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film, width_l1=cnn_widths[2], width_l2=cnn_widths[3]),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film, width_l1=cnn_widths[4], width_l2=cnn_widths[5]),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.multifc=multifc
        if multifc:
            self.win_fc = nn.Conv1d(fc_unit, fc_unit, self.win_fw+self.win_bw+1)
        self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        if self.multifc:
            fc_x = F.pad(fc_x.permute(0,2,1), (self.win_bw, self.win_fw)) # B C L 
            fc_x = self.win_fc(fc_x).transpose(1,2)
        pitchwise_x = self.pitch_linear(fc_x)
        pitchwise_x = pitchwise_x.view(batch_size, -1, self.hidden_per_pitch, 88)
        return F.relu(self.layernorm(pitchwise_x))

class MIDIFrontEnd(nn.Module):
    def __init__(self, n_per_pitch=3, detune=0.0) -> None:
        # Detune: semitone unit. 0.5 means 50 cents.
        super().__init__()
        self.midi_low = MidiSpec(1024, n_per_pitch)
        self.midi_mid = MidiSpec(4096, n_per_pitch)
        self.midi_high = MidiSpec(8192, n_per_pitch)

    def forward(self, audio, detune_list=None):
        midi_low = self.midi_low(audio, detune_list)
        midi_mid = self.midi_mid(audio, detune_list)
        midi_high = self.midi_high(audio, detune_list)
        spec = th.stack([midi_low, midi_mid, midi_high], dim=1)
        return spec # B, 3, F, T

class HarmonicCNN(nn.Module):
    def __init__(self, cnn_unit):
        super().__init__()

        self.conv  = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit, (5, 3), stride=(5,1), padding=(0, 1)),
            nn.ReLU()
        )

        map_idx = [-36, -31, -28, -24, -19, -12, 0, 12, 19, 24, 28, 31, 36]
        select_idx = []
        for n in range(88):
            select_idx.extend([n+36+el for el in map_idx])
        select_idx = th.Tensor(select_idx).to(int)
        self.register_buffer('select_idx', select_idx)
            
        self.shared = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit, (13, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn_unit, cnn_unit, (1, 1)),
            nn.ReLU())
            
        self.nonshared = nn.Sequential(
            nn.Conv2d(88*cnn_unit, 88*cnn_unit, (13, 1), groups=88),
            nn.ReLU(),
            nn.Conv2d(88*cnn_unit, 88*cnn_unit, (1, 1), groups=88),
            nn.ReLU())

    def forward(self, cnn_out):
        batch_size = cnn_out.shape[0]
        seq_len = cnn_out.shape[-1]
        cnn_unit = cnn_out.shape[1]

        cnn_out_semitone = self.conv(cnn_out) # B, C, 88, T
        x_pad = F.pad(cnn_out_semitone, (0,0,36,36))
        x_split = th.index_select(x_pad, 2, self.select_idx)  # B, C, 88*13, T
        x_split = x_split.reshape(batch_size, x_split.shape[1], 88, 13, seq_len) # B, C, 88, 13, T
        x_split = x_split.permute(0,2,1,3,4)  # B, 88, C, 13, T
    
        x_shared = self.shared(x_split.reshape(batch_size*88, cnn_unit, 13, seq_len)) # B*88, C, 1, T
        x_nonshared = self.nonshared(x_split.reshape(batch_size, 88*cnn_unit, 13, seq_len)) # B, 88*C, 1, T
        ac_out = th.concat(
            (x_shared.reshape(batch_size, 88, cnn_unit, -1), 
             x_nonshared.reshape(batch_size, 88, cnn_unit, seq_len)), dim=2) # B, 88, 2C, T
        ac_out = ac_out.permute(0, 1, 3, 2)
        return ac_out.reshape(batch_size*88, seq_len, 2*cnn_unit)  # B*88, T, 2C
        

class PAR_v2_midi(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film,
                 cnn_widths = [3,3,3,3,3,3], multifc=True):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(3, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[0], width_l2=cnn_widths[1]),
            # nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[2], width_l2=cnn_widths[3]),
            # nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[4], width_l2=cnn_widths[5]),
            nn.Dropout(0.25),
        )
        self.mapping = HarmonicCNN(cnn_unit)  # B*88, T, 2C

        '''
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        '''

        self.pitch_linear = nn.Linear(2*cnn_unit, self.hidden_per_pitch)
        # self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = self.cnn(mel)  # B C F L
        x = self.mapping(x)  # B*88 T 2C

        # fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        # pitchwise_x = self.pitch_linear(x)
        # pitchwise_x = pitchwise_x.view(batch_size, -1, self.hidden_per_pitch, 88)

        pitchwise_x = self.pitch_linear(x)
        pitchwise_x = pitchwise_x.view(batch_size, 88, -1, self.hidden_per_pitch)
        pitchwise_x = pitchwise_x.permute(0, 2, 3, 1) # B T H 88
        return F.relu(self.layernorm(pitchwise_x))

class HarmonicDilatedConv(nn.Module):
    def __init__(self, c_in, c_out, n_per_pitch=4) -> None:
        super().__init__()
        dilations = [round(12*np.log2(a)*n_per_pitch) for a in range(2, 10)]
        self.conv = nn.ModuleDict()
        for i, d in enumerate(dilations):
            self.conv[str(i)] = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, d])
    def forward(self, x):
        x = self.conv['0'](x) + self.conv['1'](x) + self.conv['2'](x) + self.conv['3'](x) + \
            self.conv['4'](x) + self.conv['5'](x) + self.conv['6'](x) + self.conv['7'](x)
        x = th.relu(x)
        return x

class PAR_v2_HPP(nn.Module):
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1]):
        if(pool_size == None):
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out),
                
            )
        else:
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out)
            )

    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film,
                 cnn_widths = [3,3,3,3,3,3], n_per_pitch=5):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.block_1 = self.get_conv2d_block(3, cnn_unit, kernel_size=7)
        self.block_2 = self.get_conv2d_block(cnn_unit, cnn_unit, kernel_size=7)
        self.block_2_5 = self.get_conv2d_block(cnn_unit, cnn_unit, kernel_size=7)

        c3_out = 128
        
        self.conv_3 = HarmonicDilatedConv(cnn_unit, c3_out, n_per_pitch)

        self.block_4 = self.get_conv2d_block(c3_out, c3_out, pool_size=[1, n_per_pitch], dilation=[1, 12*n_per_pitch])
        self.block_5 = self.get_conv2d_block(c3_out, c3_out, dilation=[1, 12])
        self.block_6 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_7 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_8 = self.get_conv2d_block(c3_out, c3_out, [5,1])

    def forward(self, mel):
        x = self.block_1(mel.permute(0, 1, 3, 2))
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)
        x = x[:,:,:,:88]
        # => [b x 1 x T x 88]

        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x
        # x = self.conv_9(x)
        # x = torch.relu(x)
        # x = self.conv_10(x)
        # x = torch.sigmoid(x)
        
        x = x.permute(0, 2, 1, 3)  # B, 128, T, 88 -> B, T, 128, 88
        return x  



class CNNTrunk(nn.Module):
    def get_conv2d_block(self, channel_in,channel_out, kernel_size = [1, 3], pool_size = None, dilation = [1, 1]):
        if(pool_size == None):
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out),
                
            )
        else:
            return nn.Sequential( 
                nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
                # nn.BatchNorm2d(channel_out),
                nn.InstanceNorm2d(channel_out)
            )

    def __init__(self, c_in = 1, c_har = 16,  embedding = 128, fixed_dilation = 24, n_per_pitch=4) -> None:
        super().__init__()

        self.block_1 = self.get_conv2d_block(c_in, c_har, kernel_size=7)
        self.block_2 = self.get_conv2d_block(c_har, c_har, kernel_size=7)
        self.block_2_5 = self.get_conv2d_block(c_har, c_har, kernel_size=7)

        c3_out = embedding
        
        self.conv_3 = HarmonicDilatedConv(c_har, c3_out, n_per_pitch)

        self.block_4 = self.get_conv2d_block(c3_out, c3_out, pool_size=[1, 4], dilation=[1, 48])
        self.block_5 = self.get_conv2d_block(c3_out, c3_out, dilation=[1, 12])
        self.block_6 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_7 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        self.block_8 = self.get_conv2d_block(c3_out, c3_out, [5,1])
        # self.conv_9 = nn.Conv2d(c3_out, 64,1)
        # self.conv_10 = nn.Conv2d(64, 1, 1)

    def forward(self, log_gram_db):
        # inputs: [b x 2 x T x n_freq] , [b x 1 x T x 88]
        # outputs: [b x T x 88]


        # img_path = 'logspecgram_preview.png'
        # if not os.path.exists(img_path):
        #     img = torch.permute(log_gram_db, [2, 0, 1]).reshape([352, 640*4]).detach().cpu().numpy()
        #     # x_grid = torchvision.utils.make_grid(x.swapaxes(0, 1), pad_value=1.0).swapaxes(0, 2).detach().cpu().numpy()
        #     # plt.imsave(img_path, (x_grid+80)/100)
        #     plt.imsave(img_path, img)

        # => [b x 1 x T x 352]
        # x = torch.unsqueeze(log_gram_db, dim=1)



        x = self.block_1(log_gram_db)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)
        # => [b x 1 x T x 88]

        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x
        # x = self.conv_9(x)
        # x = torch.relu(x)
        # x = self.conv_10(x)
        # x = torch.sigmoid(x)

        return x

class PAR_v4_midi(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film,
                 cnn_widths = [3,3,3,3,3,3], multifc=True):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(3, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[0], width_l2=cnn_widths[1]),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[2], width_l2=cnn_widths[3]),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[4], width_l2=cnn_widths[5]),
            nn.Dropout(0.25),
        )
        
        map_idx = [-36, -31, -28, -24, -19, -12, 0, 12, 19, 24, 28, 31, 36]
        self.select_idx = []
        for n in range(88):
            self.select_idx.extend([n+36+el for el in map_idx])


        self.shared = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit, (13, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn_unit, cnn_unit, (1, 1)),
            nn.ReLU())

        self.nonshared = nn.Sequential(
            nn.Conv2d(88*cnn_unit, 88*cnn_unit, (13, 1), groups=88),
            nn.ReLU(),
            nn.Conv2d(88*cnn_unit, 88*cnn_unit, (1, 1), groups=88),
            nn.ReLU())

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.multifc = multifc
        if multifc:
            self.win_fc = nn.Conv1d(fc_unit, fc_unit, self.win_fw+self.win_bw+1)
        # self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, multi_mel):
        batch_size = multi_mel.shape[0]
        x = multi_mel  # B 3 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        if self.multifc:
            fc_x = F.pad(fc_x.permute(0,2,1), (self.win_bw, self.win_fw)) # B C L 
            fc_x = self.win_fc(fc_x).transpose(1,2)
        pitchwise_x = self.pitch_linear(fc_x)
        pitchwise_x = pitchwise_x.view(batch_size, -1, self.hidden_per_pitch, 88)
        return F.relu(self.layernorm(pitchwise_x))


class PAR_v2_bn(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.win_fc = nn.Conv1d(fc_unit, fc_unit, self.win_fw+self.win_bw+1)
        self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.bn = nn.BatchNorm2d(hidden_per_pitch)

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = F.pad(fc_x.permute(0,2,1), (self.win_bw, self.win_fw)) # B C L 
        multistep_x = self.win_fc(fc_x)
        pitchwise_x = self.pitch_linear(multistep_x.transpose(1,2))
        pitchwise_x = pitchwise_x.view(batch_size, -1, self.hidden_per_pitch, 88).tranpose(1,2)
        pitchwise_x = F.relu(self.bn(pitchwise_x).transpose(1,2))
        return pitchwise_x

class PAR_v3(nn.Module):
    # SimpleConv without Pitchwise Conv
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_unit * (n_mels // 4), fc_unit),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.win_fc = nn.Conv1d(fc_unit, fc_unit, self.win_fw+self.win_bw+1)
        self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.bn = nn.BatchNorm2d(hidden_per_pitch)

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = F.pad(fc_x.permute(0,2,1), (self.win_bw, self.win_fw)) # B C L 
        multistep_x = self.win_fc(fc_x)
        pitchwise_x = self.pitch_linear(multistep_x.transpose(1,2)) # B L 88H
        pitchwise_x = pitchwise_x.view(batch_size, -1, self.hidden_per_pitch, 88).transpose(1,2) # B H L 88
        pitchwise_x = F.relu(self.bn(pitchwise_x))
        return pitchwise_x.transpose(1,2) # B L H 88
    
class PC_v8(nn.Module):
    # Compact verision of PAR_v2
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.shrink = nn.Sequential(
            nn.Conv2d(cnn_unit, 4, (1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, (1, 1)),
            nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear((n_mels // 4), self.hidden_per_pitch*88),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.win_fc = nn.Conv1d(self.hidden_per_pitch*88, self.hidden_per_pitch*88, self.win_fw+self.win_bw+1, groups=88)
        # self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.bn = nn.BatchNorm2d(hidden_per_pitch)

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        x = self.shrink(x) # B 1 F L
        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L 88H
        fc_x = F.pad(fc_x.transpose(1,2), (self.win_bw, self.win_fw)) # B 88H L 
        multistep_x = self.win_fc(fc_x)
        multistep_x = multistep_x.view(batch_size, 88, self.hidden_per_pitch, -1).transpose(1,2) # B H 88 L 
        x = F.relu(self.bn(multistep_x))
        return x.permute(0, 3, 1, 2) # B L H 88

class PC_v8_bis(nn.Module):
    # Compact verision of PAR_v2, multi-cnn fix, fc to conv1d
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.shrink = nn.Sequential(
            nn.Conv2d(cnn_unit, 4, (1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, (1, 1)),
            nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Conv1d((n_mels // 4), self.hidden_per_pitch*88, 1),
            nn.Dropout(0.25),
            nn.ReLU()
        )
        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.win_fc = nn.Conv2d(self.hidden_per_pitch, self.hidden_per_pitch, (1, self.win_fw+self.win_bw+1))
        # self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)
        self.bn = nn.BatchNorm2d(hidden_per_pitch)

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        x = self.shrink(x) # B 1 F L
        fc_x = self.fc(x.squeeze(1)) # B 88H L
        fc_x = fc_x.reshape(batch_size, 88, self.hidden_per_pitch, -1) # B 88 H L
        fc_x = F.pad(fc_x, (self.win_bw, self.win_fw)) # B 88 H L 
        multistep_x = self.win_fc(fc_x.transpose(1,2)) # B H 88 L 
        multistep_x = multistep_x # B H 88 L 
        x = F.relu(self.bn(multistep_x))
        return x.permute(0, 3, 1, 2) # B L H 88

class PC_v9(nn.Module):
    # Compact verision of PAR_v2
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film,
                 cnn_widths = [3,3,3,3,3,3], multifc=True):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        self.multifc = multifc
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film, width_l1=cnn_widths[0], width_l2=cnn_widths[1]),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film, width_l1=cnn_widths[2], width_l2=cnn_widths[3]),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film, width_l1=cnn_widths[4], width_l2=cnn_widths[5]),
            nn.Dropout(0.25),
        )


        self.shrink = nn.Sequential(
            nn.Conv2d(cnn_unit, 4, (1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, (1, 1)),
            nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Conv1d((n_mels // 4), self.hidden_per_pitch*88, 1),
            nn.BatchNorm1d(self.hidden_per_pitch*88),
            nn.Dropout(0.25),
            nn.ReLU()
        )

        self.fc_1 = nn.Sequential(
            nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88),
            nn.BatchNorm1d(self.hidden_per_pitch*88),
            nn.ReLU()
        ) 
        self.fc_2 = nn.Sequential(
            nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88),
            nn.BatchNorm1d(self.hidden_per_pitch*88),
            nn.ReLU()
        ) 
        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        if multifc:
            self.win_fc = nn.Conv2d(self.hidden_per_pitch, self.hidden_per_pitch, (1, self.win_fw+self.win_bw+1))
        # self.pitch_linear = nn.Linear(fc_unit, self.hidden_per_pitch*88)

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        x = self.shrink(x) # B 1 F L
        fc_x = self.fc(x.squeeze(1)) # B 88H L
        fc_x = self.fc_1(fc_x)
        fc_x = self.fc_2(fc_x)
        fc_x = fc_x.reshape(batch_size, self.hidden_per_pitch, 88, -1)
        if self.multifc:
            fc_x = F.pad(fc_x, (self.win_bw, self.win_fw)) # B H 88 L 
            fc_x = self.win_fc(fc_x)
        return fc_x.permute(0, 3, 1, 2) # B L H 88
        
class PC_v4(nn.Module):
    # two fc path, one with large conv. Channel last
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film, shrink_channels=[4,1]):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 * frames * 700
        
        self.cnn0 = nn.Sequential(
            nn.Conv2d(1, cnn_unit, (5, 5), padding=2),
            nn.MaxPool2d((1, 4)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn0.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn0.append(nn.ReLU())

        self.cnn1 = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit//2, (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit//2),
        )
        if use_film:
            self.cnn1.append(FilmLayer(n_mels//4, cnn_unit//2, hidden=16))
        self.cnn1.append(nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(cnn_unit//2, cnn_unit//4, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(cnn_unit//4),
        )
        if use_film:
            self.cnn2.append(FilmLayer(n_mels//4, cnn_unit//4, hidden=16))
        self.cnn2.append(nn.ReLU())

        self.large_conv = nn.Sequential(
            nn.ZeroPad2d((29,30, 0, 0)),
            nn.Conv2d(cnn_unit//4, cnn_unit//4, (1, 60)),
            nn.BatchNorm2d(cnn_unit//4),
        )
        if use_film:
            self.large_conv.append(FilmLayer(n_mels//4, cnn_unit//4, hidden=16))
        self.large_conv.append(nn.ReLU())
        
        s_channels = shrink_channels
        self.shrink_conv_path0 = nn.Sequential(
            nn.Conv2d(cnn_unit//4, s_channels[0], (1, 1)),
            nn.BatchNorm2d(s_channels[0]),
            nn.ReLU(),
            nn.Conv2d(s_channels[0], s_channels[1], (1, 1)),
            nn.ReLU()
            )

        self.pitch_fc_path0 = nn.Linear(n_mels//4*s_channels[1], 88*hidden_per_pitch//2)

        self.shrink_conv_path1 = nn.Sequential(
            nn.Conv2d(cnn_unit//4, s_channels[0], (1, 1)),
            nn.BatchNorm2d(s_channels[0]),
            nn.ReLU(),
            nn.Conv2d(s_channels[0], s_channels[1], (1, 1)),
            nn.ReLU()
            )
        self.pitch_fc_path1 = nn.Linear(n_mels//4*s_channels[1], 88*hidden_per_pitch//2)
        self.multi_conv = nn.Sequential(
            nn.ZeroPad2d((0,0, win_bw, win_fw)),
            nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (win_bw+win_fw+1,1))
        )
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])
            

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = self.cnn0(x)  # B C L F
        x = self.cnn1(x)
        x = self.cnn2(x)

        x_0 = self.large_conv(x)
        x_0 = self.shrink_conv_path0(x_0)
        x_0 = x_0.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_0 = self.pitch_fc_path0(x_0)  # B L 88H
        x_0 = x_0.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x_1 = self.shrink_conv_path1(x)
        x_1 = x_1.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_1 = self.pitch_fc_path1(x_1)  # B L 88H
        x_1 = x_1.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x_cat = th.cat([x_0, x_1], dim=2).permute(0, 2, 1, 3) # B H L 88
        x_cat = self.multi_conv(x_cat).transpose(1,2) # B L H 88
        x_cat = self.layernorm(x_cat)
        x_cat = F.relu(x_cat) # B L H 88

        return x_cat

class PC_v4_v2(nn.Module):
    # PC_v4, but no last multi-conv
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film, shrink_channels=[4,1]):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 * frames * 700
        
        self.cnn0 = nn.Sequential(
            nn.Conv2d(1, cnn_unit, (5, 5), padding=2),
            nn.MaxPool2d((1, 4)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn0.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn0.append(nn.ReLU())

        self.cnn1 = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit//2, (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit//2),
        )
        if use_film:
            self.cnn1.append(FilmLayer(n_mels//4, cnn_unit//2, hidden=16))
        self.cnn1.append(nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(cnn_unit//2, cnn_unit//4, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(cnn_unit//4),
        )
        if use_film:
            self.cnn2.append(FilmLayer(n_mels//4, cnn_unit//4, hidden=16))
        self.cnn2.append(nn.ReLU())

        self.large_conv = nn.Sequential(
            nn.ZeroPad2d((29,30, 0, 0)),
            nn.Conv2d(cnn_unit//4, cnn_unit//4, (win_fw+win_bw+1, 60)),
            nn.BatchNorm2d(cnn_unit//4),
        )
        if use_film:
            self.large_conv.append(FilmLayer(n_mels//4, cnn_unit//4, hidden=16))
        self.large_conv.append(nn.ReLU())
        
        s_channels = shrink_channels
        self.shrink_conv_path0 = nn.Sequential(
            nn.Conv2d(cnn_unit//4, s_channels[0], (1, 1)),
            nn.BatchNorm2d(s_channels[0]),
            nn.ReLU(),
            nn.Conv2d(s_channels[0], s_channels[1], (1, 1)),
            nn.ReLU()
            )

        self.pitch_fc_path0 = nn.Linear(n_mels//4*s_channels[1], 88*hidden_per_pitch//2)

        self.shrink_conv_path1 = nn.Sequential(
            nn.Conv2d(cnn_unit//4, s_channels[0], (1, 1)),
            nn.BatchNorm2d(s_channels[0]),
            nn.ReLU(),
            nn.Conv2d(s_channels[0], s_channels[1], (1, 1)),
            nn.ReLU()
            )
        self.pitch_fc_path1 = nn.Linear(n_mels//4*s_channels[1], 88*hidden_per_pitch//2)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])
            

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = self.cnn0(x)  # B C L F
        x = self.cnn1(x)
        x = self.cnn2(x)

        x_0 = self.large_conv(x)
        x_0 = self.shrink_conv_path0(x_0)
        x_0 = x_0.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_0 = self.pitch_fc_path0(x_0)  # B L 88H
        x_0 = x_0.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x_1 = self.shrink_conv_path1(x)
        x_1 = x_1.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_1 = self.pitch_fc_path1(x_1)  # B L 88H
        x_1 = x_1.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x_cat = th.cat([x_0, x_1], dim=2).permute(0, 2, 1, 3).transpose(1,2) # B H L 88
        x_cat = self.layernorm(x_cat)
        x_cat = F.relu(x_cat) # B L H 88

        return x_cat

class PC_v5(nn.Module):
    # similar to PC_v4, but similar architecture as the original, but fc path does not have multistep_fc
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film, shrink_channels=[4,1]):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 * frames * 700
        
        self.cnn0 = nn.Sequential(
            nn.Conv2d(1, cnn_unit, (5, 5), padding=2),
            nn.MaxPool2d((1, 4)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn0.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn0.append(nn.ReLU())

        self.cnn1 = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=(1,1)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn1.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn1.append(nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn2.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn2.append(nn.ReLU())

        self.large_conv = nn.Sequential(
            nn.ZeroPad2d((29,30, 0, 0)),
            nn.Conv2d(cnn_unit, hidden_per_pitch//2, (1, 60)),
            nn.BatchNorm2d(hidden_per_pitch//2),
        )
        if use_film:
            self.large_conv.append(FilmLayer(n_mels//4, hidden_per_pitch//2, hidden=16))
        self.large_conv.append(nn.ReLU())
        
        self.pitch_fc_path0 = nn.Linear(n_mels//4*hidden_per_pitch//2, 88*hidden_per_pitch//2)

        self.pitch_fc1_path1 = nn.Linear(n_mels//4*cnn_unit, 768)
        self.pitch_fc2_path1 = nn.Linear(768, 88*hidden_per_pitch//2)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])
            

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = self.cnn0(x)  # B C L F
        x = self.cnn1(x)
        x = self.cnn2(x)

        x_0 = self.large_conv(x)
        x_0 = x_0.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_0 = self.pitch_fc_path0(x_0)  # B L 88H
        x_0 = x_0.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x_1 = x.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_1 = self.pitch_fc1_path1(x_1)  # B L 768
        x_1 = self.pitch_fc2_path1(x_1)  # B L 768
        x_1 = x_1.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x_cat = th.cat([x_0, x_1], dim=2).permute(0, 2, 1, 3) # B H L 88
        x_cat = self.layernorm(x_cat.transpose(1,2))
        x_cat = F.relu(x_cat) # B L H 88

        return x_cat

class PC_v6(nn.Module):
    # v4 variant, but with one pathway, and no mulit-step at the last.
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film, shrink_channels=[4,1]):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 * frames * 700
        
        self.cnn0 = nn.Sequential(
            nn.Conv2d(1, cnn_unit, (5, 5), padding=2),
            nn.MaxPool2d((1, 4)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn0.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn0.append(nn.ReLU())

        self.cnn1 = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit//2, (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit//2),
        )
        if use_film:
            self.cnn1.append(FilmLayer(n_mels//4, cnn_unit//2, hidden=16))
        self.cnn1.append(nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(cnn_unit//2, cnn_unit//4, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(cnn_unit//4),
        )
        if use_film:
            self.cnn2.append(FilmLayer(n_mels//4, cnn_unit//4, hidden=16))
        self.cnn2.append(nn.ReLU())

        self.large_conv = nn.Sequential(
            nn.ZeroPad2d((29,30, 0, 0)),
            nn.Conv2d(cnn_unit//4, cnn_unit//4, (1, 60)),
            nn.BatchNorm2d(cnn_unit//4),
        )
        if use_film:
            self.large_conv.append(FilmLayer(n_mels//4, cnn_unit//4, hidden=16))
        self.large_conv.append(nn.ReLU())
        
        s_channels = shrink_channels
        self.shrink_conv_path0 = nn.Sequential(
            nn.Conv2d(cnn_unit//4, s_channels[0], (1, 1)),
            nn.BatchNorm2d(s_channels[0]),
            nn.ReLU(),
            nn.Conv2d(s_channels[0], s_channels[1], (1, 1)),
            nn.ReLU()
            )

        self.pitch_fc_path0 = nn.Linear(n_mels//4*s_channels[1], 88*hidden_per_pitch)

        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])
            

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = self.cnn0(x)  # B C L F
        x = self.cnn1(x)
        x = self.cnn2(x)

        x_0 = self.large_conv(x)
        x_0 = self.shrink_conv_path0(x_0)
        x_0 = x_0.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_0 = self.pitch_fc_path0(x_0)  # B L 88H
        x_0 = x_0.view(batch_size, -1, self.hidden_per_pitch, 88)
        x_cat = self.layernorm(0)
        x_cat = F.relu(x_cat) # B L H 88

        return x_cat

class PC_v7(nn.Module):
    # simiar to PC_v4, but original conv stacks, no mulit-step at the last.
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film, shrink_channels=[4,1]):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 * frames * 700
        
        self.cnn0 = nn.Sequential(
            nn.Conv2d(1, cnn_unit, (5, 5), padding=2),
            nn.MaxPool2d((1, 4)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn0.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn0.append(nn.ReLU())

        self.cnn1 = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit, (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn1.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn1.append(nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(cnn_unit, cnn_unit, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.cnn2.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.cnn2.append(nn.ReLU())

        self.large_conv = nn.Sequential(
            nn.ZeroPad2d((59, 0, win_bw, win_fw)),
            nn.Conv2d(cnn_unit, cnn_unit, (win_fw+win_bw+1, 60)),
            nn.BatchNorm2d(cnn_unit),
        )
        if use_film:
            self.large_conv.append(FilmLayer(n_mels//4, cnn_unit, hidden=16))
        self.large_conv.append(nn.ReLU())
        
        s_channels = shrink_channels
        self.shrink_conv_path0 = nn.Sequential(
            nn.Conv2d(cnn_unit, s_channels[0], (1, 1)),
            nn.BatchNorm2d(s_channels[0]),
            nn.ReLU(),
            nn.Conv2d(s_channels[0], s_channels[1], (1, 1)),
            nn.ReLU()
            )

        self.pitch_fc_path0 = nn.Linear(n_mels//4*s_channels[1], 88*hidden_per_pitch//2)

        self.shrink_conv_path1 = nn.Sequential(
            nn.Conv2d(cnn_unit, s_channels[0], (1, 1)),
            nn.BatchNorm2d(s_channels[0]),
            nn.ReLU(),
            nn.Conv2d(s_channels[0], s_channels[1], (1, 1)),
            nn.ReLU()
            )
        self.pitch_fc_path1 = nn.Linear(n_mels//4*s_channels[1], 88*hidden_per_pitch//2)
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])
            

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = self.cnn0(x)  # B C L F
        x = self.cnn1(x)
        x = self.cnn2(x)

        x_0 = self.large_conv(x)
        x_0 = self.shrink_conv_path0(x_0)
        x_0 = x_0.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_0 = self.pitch_fc_path0(x_0)  # B L 88H
        x_0 = x_0.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x_1 = self.shrink_conv_path1(x)
        x_1 = x_1.permute(0, 2, 1, 3).flatten(-2) # B L C
        x_1 = self.pitch_fc_path1(x_1)  # B L 88H
        x_1 = x_1.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x_cat = th.cat([x_0, x_1], dim=2).permute(0, 2, 1, 3).transpose(1,2) # B L H 88
        x_cat = self.layernorm(x_cat)
        x_cat = F.relu(x_cat) # B L H 88

        return x_cat

class PAR_CQT(nn.Module):
    # large conv - pitchwise fc model
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        # self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)
        self.large_conv = nn.Conv2d(cnn_unit, hidden_per_pitch, (49, self.win_fw+self.win_bw+1),
                                    stride=(2, 1))

        self.fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L
        x = F.pad(x, (self.win_bw, self.win_fw, 24, 3))
        x = self.large_conv(x) # B C 88, L
        x = x.view(x.shape[0], self.hidden_per_pitch*88, -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = x.view(x.shape[0], self.hidden_per_pitch, 88, -1).permute(0, 3, 1, 2)

        return F.relu(self.layernorm(x))

class PAR_Mel2(nn.Module):
    # three branch, use semitone-spaced middle feature for both branches
    def __init__(self, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn_low = nn.Sequential(
            FilmBlock(1, cnn_unit, 57, use_film=use_film),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, 19, use_film=use_film),
            FilmBlock(cnn_unit, cnn_unit, 19, use_film=use_film),
        )
        self.cnn_high = nn.Sequential(
            FilmBlock(1, cnn_unit, 472, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, 472//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, 472//4, use_film=use_film),
            nn.Dropout(0.25),
            nn.MaxPool2d((2, 1)),
        )
        self.cnn_lowest = nn.Sequential(
            FilmBlock(1, cnn_unit, 11, use_film=use_film),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, 11, use_film=use_film),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, 11, use_film=use_film),
        )

        self.large_conv = nn.Conv2d(cnn_unit, hidden_per_pitch, (61, self.win_fw+self.win_bw+1))

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * 89, fc_unit),  # 89=19+59+11
            nn.Dropout(0.25),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_unit, hidden_per_pitch*88),
            nn.ReLU()
        )
        self.group_conv1 = nn.Conv1d(hidden_per_pitch*2*88, hidden_per_pitch*2*88, 1, padding=0, groups=88)
        self.group_conv2 = nn.Conv1d(hidden_per_pitch*2*88, hidden_per_pitch*2*88, 1, padding=0, groups=88)
        self.group_conv3 = nn.Conv1d(hidden_per_pitch*2*88, hidden_per_pitch*2*88, 1, padding=0, groups=88)

        self.layernorm = nn.LayerNorm([hidden_per_pitch*2, 88])

    def forward(self, mel_low, mel_high, spec):
        batch_size = mel_low.shape[0]
        # lowest freq
        lowest = spec.unsqueeze(1)  # B 1 L F
        lowest = lowest.transpose(2,3)  # B 1 F L
        lowest = self.cnn_lowest(lowest)
        # middle
        low = mel_low.unsqueeze(1)  # B 1 L F
        low = low.transpose(2,3)  # B 1 F L
        low = self.cnn_low(low)  # B C F L
        # do the same on high
        high = mel_high.unsqueeze(1)  # B 1 L F
        high = high.transpose(2,3)  # B 1 F L
        high = self.cnn_high(high)  # B C F L

        conv_feature = th.cat([low, high], dim=2)
        fc_feature = th.cat([lowest, low, high], dim=2)

        conv_x = F.pad(conv_feature, (self.win_bw, self.win_fw, 32, 88+28-78))
        conv_x = self.large_conv(conv_x) # B H 88, L
        conv_x = conv_x.view(conv_x.shape[0], self.hidden_per_pitch*88, -1)

        fc_x = self.fc(fc_feature.permute(0, 3, 1, 2).flatten(-2))
        fc_x = self.fc2(fc_x).transpose(1, 2) # B 88H L

        x = th.cat([conv_x, fc_x], dim=1)
        x = F.relu(self.group_conv1(x))
        x = F.relu(self.group_conv2(x))
        x = F.relu(self.group_conv3(x))
        x = x.view(x.shape[0], self.hidden_per_pitch*2, 88, -1).permute(0, 3, 1, 2)

        return F.relu(self.layernorm(x)), fc_feature.detach()

class PAR_Mel2_highres(nn.Module):
    # three branch, use sub-semitone-spaced middle feature for fc
    def __init__(self, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn_low = nn.Sequential(
            FilmBlock(1, cnn_unit, 57, use_film=use_film),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, 19, use_film=use_film),
            FilmBlock(cnn_unit, cnn_unit, 19, use_film=use_film),
        )
        self.cnn_high = nn.Sequential(
            FilmBlock(1, cnn_unit, 472, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, 472//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, 472//4, use_film=use_film),
            nn.Dropout(0.25),
        )
        self.cnn_lowest = nn.Sequential(
            FilmBlock(1, cnn_unit, 11, use_film=use_film),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, 11, use_film=use_film),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, 11, use_film=use_film),
        )

        self.large_conv = nn.Conv2d(cnn_unit, hidden_per_pitch, (61, self.win_fw+self.win_bw+1))

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * 148, fc_unit),  # 19+59*2+11 = 148
            nn.Dropout(0.25),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_unit, hidden_per_pitch*88),
            nn.ReLU()
        )
        self.group_conv1 = nn.Conv1d(hidden_per_pitch*2*88, hidden_per_pitch*2*88, 1, padding=0, groups=88)
        self.group_conv2 = nn.Conv1d(hidden_per_pitch*2*88, hidden_per_pitch*2*88, 1, padding=0, groups=88)
        self.group_conv3 = nn.Conv1d(hidden_per_pitch*2*88, hidden_per_pitch*2*88, 1, padding=0, groups=88)

        self.layernorm = nn.LayerNorm([hidden_per_pitch*2, 88])

    def forward(self, mel_low, mel_high, spec):
        batch_size = mel_low.shape[0]
        # lowest freq
        lowest = spec.unsqueeze(1)  # B 1 L F
        lowest = lowest.transpose(2,3)  # B 1 F L
        lowest = self.cnn_lowest(lowest)
        # middle
        low = mel_low.unsqueeze(1)  # B 1 L F
        low = low.transpose(2,3)  # B 1 F L
        low = self.cnn_low(low)  # B C F L
        # do the same on high
        high = mel_high.unsqueeze(1)  # B 1 L F
        high = high.transpose(2,3)  # B 1 F L
        high = self.cnn_high(high)  # B C F L

        conv_feature = th.cat([low, F.max_pool2d(high, (2,1))], dim=2)
        fc_feature = th.cat([lowest, low, high], dim=2)

        conv_x = F.pad(conv_feature, (self.win_bw, self.win_fw, 32, 88+28-78))
        conv_x = self.large_conv(conv_x) # B H 88, L
        conv_x = conv_x.view(conv_x.shape[0], self.hidden_per_pitch*88, -1)

        fc_x = self.fc(fc_feature.permute(0, 3, 1, 2).flatten(-2))
        fc_x = self.fc2(fc_x).transpose(1, 2) # B 88H L

        x = th.cat([conv_x, fc_x], dim=1)
        x = F.relu(self.group_conv1(x))
        x = F.relu(self.group_conv2(x))
        x = F.relu(self.group_conv3(x))
        x = x.view(x.shape[0], self.hidden_per_pitch*2, 88, -1).permute(0, 3, 1, 2)

        return F.relu(self.layernorm(x)), fc_feature.detach()

class PAR_CQT_v2(nn.Module):
    # two-path model
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.large_conv = nn.Conv2d(cnn_unit, hidden_per_pitch, (49, self.win_fw+self.win_bw+1),
                                    stride=(2, 1))
        self.pitch_fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.pitch_fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.pitch_fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.fc_1 = nn.Linear(790//4*cnn_unit, fc_unit)
        self.fc_2 = nn.Linear(fc_unit, hidden_per_pitch*88)

        self.layernorm = nn.LayerNorm([hidden_per_pitch*2, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        cnn_out = self.cnn(x)  # B C F L
        x = F.pad(cnn_out, (self.win_bw, self.win_fw, 24, 3))
        x = self.large_conv(x) # B H 88, L

        x_1 = x.view(batch_size, self.hidden_per_pitch*88, -1)
        x_1 = F.relu(self.pitch_fc_1(x_1))
        x_1 = F.relu(self.pitch_fc_2(x_1))
        x_1 = F.relu(self.pitch_fc_3(x_1))
        x_1 = x_1.view(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 3, 1, 2)

        x_2 = self.fc_1(cnn_out.permute(0,3,1,2).flatten(-2))
        x_2 = self.fc_2(x_2).view(batch_size, -1, self.hidden_per_pitch, 88)
        x = th.cat((x_1, x_2), -2)

        return F.relu(self.layernorm(x))

class PAR_CQT_v3(nn.Module):
    # two-path model
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * 700
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.pitch_cnn1 = nn.Conv2d(cnn_unit, hidden_per_pitch, (49, self.win_fw+self.win_bw+1),
                                    stride=(2,1))
        self.pitch_cnn2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (49, 1))
        self.pitch_film1 = FilmLayer(88, hidden_per_pitch)
        self.pitch_film2 = FilmLayer(88, hidden_per_pitch)

        self.fc_1 = nn.Linear(790//4*cnn_unit, fc_unit)
        self.fc_2 = nn.Linear(fc_unit, hidden_per_pitch*88)

        self.layernorm = nn.LayerNorm([hidden_per_pitch*2, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        cnn_out = self.cnn(x)  # B C F L
        x = F.pad(cnn_out, (self.win_bw, self.win_fw, 24, 3))
        x = self.pitch_cnn1(x) # B H 88, L
        x = self.pitch_film1(x.transpose(2,3)).transpose(2,3)
        x = F.pad(x, (0, 0, 24, 24)) 
        x = self.pitch_cnn2(x)
        x = self.pitch_film2(x.transpose(2,3)) # B H L F 
        x = x.permute(0, 2, 1, 3)

        x_2 = self.fc_1(cnn_out.permute(0,3,1,2).flatten(-2))
        x_2 = self.fc_2(x_2).view(batch_size, -1, self.hidden_per_pitch, 88)
        x = th.cat((x, x_2), -2)

        return F.relu(self.layernorm(x))

class AllConv(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit, win_fw, win_bw, hidden_per_pitch, use_film):
        super().__init__()

        self.win_fw = win_fw
        self.win_bw = win_bw
        self.hidden_per_pitch = hidden_per_pitch
        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            FilmBlock(1, cnn_unit, n_mels, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            FilmBlock(cnn_unit, cnn_unit, n_mels//2, use_film=use_film),
            nn.MaxPool2d((2, 1)),
            FilmBlock(cnn_unit, cnn_unit, n_mels//4, use_film=use_film),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Linear((cnn_unit) * (n_mels // 4), fc_unit),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.win_fc = nn.Linear(fc_unit*(win_fw+win_bw+1), hidden_per_pitch//2*88)

        self.pitch_cnn1 = nn.Conv2d(cnn_unit, hidden_per_pitch//2, (60, self.win_fw+self.win_bw+1))
        self.layernorm = nn.LayerNorm([hidden_per_pitch, 88])

    def forward(self, mel):
        batch_size = mel.shape[0]
        x = mel.unsqueeze(1)  # B 1 L F
        x = x.transpose(2,3)  # B 1 F L
        x = self.cnn(x)  # B C F L

        fc_x = self.fc(x.permute(0, 3, 1, 2).flatten(-2)) # B L C
        fc_x = fc_x.transpose(1,2)  # B C L
        fc_x = F.pad(fc_x, (self.win_bw, self.win_fw)).unsqueeze(3)
        fc_x = F.unfold(fc_x, (self.win_bw + self.win_fw + 1, 1)) 
        fc_x = self.win_fc(fc_x.transpose(1,2))  # B L C
        fc_x = fc_x.view(batch_size, -1, self.hidden_per_pitch//2, 88)

        x = F.pad(x, (self.win_bw, self.win_fw, 59, 0))
        x = self.pitch_cnn1(x)[:,:,:88,:]
        x = x.permute(0, 3, 1, 2)

        x = th.cat((x, fc_x), -2)
        x = self.layernorm(x)
        return F.relu(x)

class PC(nn.Module):
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film=True, v2=False):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw
        if v2:
            cnn_multipler = [4, 2, 1]
        else:
            cnn_multipler = [1, 1, 1]

        # input is batch_size * 1 channel * frames * input_features
        self.use_film = use_film
        if use_film:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                FilmLayer(n_mels, cnn_unit*cnn_multipler[0], hidden=16),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[1], hidden=16),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[2], hidden=16),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                nn.ReLU(),
            )
        f_size = 40 
        self.large_conv_l1 = nn.Conv2d(cnn_unit*cnn_multipler[2], hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        if use_film:
            self.film_1 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)
            self.film_2 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)
            self.film_3 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)

        self.fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        # mel : B L F
        if self.use_film:
            x = mel.unsqueeze(1)
            batch_size = x.shape[0]
            x = self.cnn(x) # (B x H x L x n_mels/4)
            x = self.large_conv_l1(F.pad(x, (0,39)))
            x = F.relu(self.film_1(x))
            x = self.large_conv_l2(F.pad(x, (0,39)))
            x = F.relu(self.film_2(x))
            x = self.large_conv_l3(F.pad(x, (0,39)))
            x = F.relu(self.film_3(x))[:,:,:,:88]
        else:
            x = mel.unsqueeze(1)
            batch_size = x.shape[0]
            x = self.cnn(x) # (B x H x L x n_mels/4)
            x = self.large_conv_l1(F.pad(x, (0,39)))
            x = F.relu(x)
            x = self.large_conv_l2(F.pad(x, (0,39)))
            x = F.relu(x)
            x = self.large_conv_l3(F.pad(x, (0,39)))
            x = F.relu(x)[:,:,:,:88]
            
        x = x.transpose(2,3).reshape(batch_size, self.hidden_per_pitch*88, -1)
        x = self.fc_1(x)  # B x 1 x L x n_mels/4
        res = self.fc_2(F.relu(x))
        res = self.fc_3(F.relu(res))
        x = x + res
        x = x.reshape(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 1, 3, 2)
        # B, H, L, 88

        x = self.window_cnn(x)  # B x H x L x 88
        x = x.transpose(1, 2)
        # x = x.flatten(-2)

        return x  # B x L x H x 88

class PC_v3(nn.Module):
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film=True, v2=False):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw
        if v2:
            cnn_multipler = [4, 2, 1]
        else:
            cnn_multipler = [1, 1, 1]

        # input is batch_size * 1 channel * frames * input_features
        self.use_film = use_film
        if use_film:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                FilmLayer(n_mels, cnn_unit*cnn_multipler[0], hidden=16),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[1], hidden=16),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[2], hidden=16),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                nn.ReLU(),
            )
        f_size = 40 
        self.large_conv_l1 = nn.Conv2d(cnn_unit*cnn_multipler[2], hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        if use_film:
            self.film_1 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)
            self.film_2 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)
            self.film_3 = FilmLayer(n_mels//4, hidden_per_pitch, hidden=16)

        self.fc_1 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        if self.use_film:
            x = mel.unsqueeze(1)
            batch_size = x.shape[0]
            x = self.cnn(x) # (B x H x L x n_mels/4)
            x = self.large_conv_l1(F.pad(x, (0,39)))
            x = F.relu(self.film_1(x))
            x = self.large_conv_l2(F.pad(x, (0,39)))
            x = F.relu(self.film_2(x))
            x = self.large_conv_l3(F.pad(x, (0,39)))
            x = F.relu(self.film_3(x))[:,:,:,:88]
        else:
            x = mel.unsqueeze(1)
            batch_size = x.shape[0]
            x = self.cnn(x) # (B x H x L x n_mels/4)
            x = self.large_conv_l1(F.pad(x, (0,39)))
            x = F.relu(x)
            x = self.large_conv_l2(F.pad(x, (0,39)))
            x = F.relu(x)
            x = self.large_conv_l3(F.pad(x, (0,39)))
            x = F.relu(x)[:,:,:,:88]
            
        x = x.transpose(2,3).reshape(batch_size, self.hidden_per_pitch*88, -1)
        x = self.fc_1(x)  # B x 1 x L x n_mels/4
        res = self.fc_2(F.relu(x))
        res = self.fc_3(F.relu(res))
        x = x + res
        x = x.reshape(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 1, 3, 2)
        # B, H, L, 88

        x = self.window_cnn(x)  # B x H x L x 88
        x = x.transpose(1, 2)
        # x = x.flatten(-2)

        return x  # B x L x H x 88



class PC_CQT(nn.Module):
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film=True):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw
        cnn_multipler = [4, 2, 1]

        # input is batch_size * 1 channel * frames * input_features
        self.use_film = use_film
        if use_film:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                FilmLayer(n_mels, cnn_unit*cnn_multipler[0], hidden=16),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[1], hidden=16),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                FilmLayer(n_mels//4, cnn_unit*cnn_multipler[2], hidden=16),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                # layer 0
                nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),

                nn.Dropout(0.25),

                nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
                nn.ReLU(),
                nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
                nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
                nn.ReLU(),
            )
        f_size = 49 
        self.large_conv_l1 = nn.Conv2d(cnn_unit*cnn_multipler[2], hidden_per_pitch, (1, f_size), padding=0, stride=(1,2))
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, f_size), padding=0)
        if use_film:
            self.film_1 = FilmLayer(88, hidden_per_pitch, hidden=16)
            self.film_2 = FilmLayer(88, hidden_per_pitch, hidden=16)
            self.film_3 = FilmLayer(88, hidden_per_pitch, hidden=16)
       
        self.fc_0 = nn.Sequential(
            nn.Conv2d(cnn_unit*cnn_multipler[2], 4, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1, padding=0),
            nn.ReLU(),
        )
        self.fc_1 = nn.Conv1d(790//4, hidden_per_pitch*88, 1, padding=0)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch*2, hidden_per_pitch*2, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        batch_size = x.shape[0]
        cnn_out = self.cnn(x) # (B x H x L x n_mels/4)
        if self.use_film:
            x = self.large_conv_l1(F.pad(cnn_out, (24,3)))
            x = F.relu(self.film_1(x))
            x = self.large_conv_l2(F.pad(x, (24,24)))
            x = F.relu(self.film_2(x))
            x = self.large_conv_l3(F.pad(x, (24,24)))
            x = F.relu(self.film_3(x))
        else:
            x = self.large_conv_l1(F.pad(cnn_out, (24,3)))
            x = F.relu(x)
            x = self.large_conv_l2(F.pad(x, (24,24)))
            x = F.relu(x)
            x = self.large_conv_l3(F.pad(x, (24,24)))
            x = F.relu(x)
        cnn_out = cnn_out.contiguous()
        x_pitchwise = self.fc_0(cnn_out)  # B x 1 x L x n_mels/4
        x_pitchwise = x_pitchwise.transpose(1, 2).flatten(-2).transpose(1,2)# (B x n_mels/4 x L)

        x_pitchwise = F.relu(self.fc_1(x_pitchwise)) # B x H*88 x L
        res = self.fc_2(x_pitchwise)
        res = self.fc_3(F.relu(res))
        x_pitchwise = x_pitchwise + res
        x_pitchwise = x_pitchwise.reshape(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 1, 3, 2)
        # B, H, L, 88

        x = th.cat((x_pitchwise, x), 1)
        x = self.window_cnn(x)  # B x H x L x 88
        x = x.transpose(1, 2)
        # x = x.flatten(-2)

        return x  # B x L x H x 88

class PC_CQT_v2(nn.Module):
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film=True):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw
        cnn_multipler = [4, 2, 1]

        # input is batch_size * 1 channel * frames * input_features
        self.use_film = use_film
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
            nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),

            nn.Dropout(0.25),

            nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
            nn.ReLU(),
            nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
            nn.ReLU(),
        )
        f_size = 88 
        self.large_conv_l1 = nn.Conv2d(cnn_unit*cnn_multipler[2], hidden_per_pitch, (1, 88), padding=0)
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, 1), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, 1), padding=0)
        if use_film:
            self.film_1 = FilmLayer(88, hidden_per_pitch, hidden=16)
            self.film_2 = FilmLayer(88, hidden_per_pitch, hidden=16)
            self.film_3 = FilmLayer(88, hidden_per_pitch, hidden=16)
       
        self.fc_0 = nn.Sequential(
            nn.Conv2d(cnn_unit*cnn_multipler[2], 4, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1, padding=0),
            nn.ReLU(),
        )
        self.fc_1 = nn.Conv1d(790//4, hidden_per_pitch*88, 1, padding=0)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch*2, hidden_per_pitch*2, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        batch_size = x.shape[0]
        cnn_out = self.cnn(x) # (B x H x L x n_mels/4)
        cnn_out = cnn_out.contiguous()
        # print(f'cnn_out:{cnn_out.shape}')
        if self.use_film:
            x = self.large_conv_l1(cnn_out)[:,:,:,:88]
            # print(f'x:{x.shape}')
            x = F.relu(self.film_1(x))
            x = self.large_conv_l2(x)
            x = F.relu(self.film_2(x))
            x = self.large_conv_l3(x)
            x = F.relu(self.film_3(x))
        else:
            x = self.large_conv_l1(cnn_out)[:,:,:,:88]
            x = F.relu(x)
            x = self.large_conv_l2(x)
            x = F.relu(x)
            x = self.large_conv_l3(x)
            x = F.relu(x)
        x_pitchwise = self.fc_0(cnn_out)  # B x 1 x L x n_mels/4
        x_pitchwise = x_pitchwise.transpose(1, 2).flatten(-2).transpose(1,2)  # (B x n_mels/4 x L)

        x_pitchwise = F.relu(self.fc_1(x_pitchwise)) # B x H*88 x L
        res = self.fc_2(x_pitchwise.contiguous())
        res = self.fc_3(F.relu(res.contiguous()))
        x_pitchwise = x_pitchwise + res
        x_pitchwise = x_pitchwise.reshape(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 1, 3, 2)
        # B, H, L, 88

        x = th.cat((x_pitchwise, x), 1)
        x = self.window_cnn(x)  # B x H x L x 88
        x = x.transpose(1, 2)
        # x = x.flatten(-2)

        return x  # B x L x H x 88

class PC_CQT_v3(nn.Module):
    # frequency match version
    def __init__(self, n_mels, cnn_unit, win_fw, win_bw, hidden_per_pitch, use_film=True):
        super().__init__()

        self.n_mels = n_mels
        self.hidden_per_pitch = hidden_per_pitch
        self.win_bw = win_bw
        self.win_fw = win_fw
        cnn_multipler = [4, 2, 1]

        # input is batch_size * 1 channel * frames * input_features
        self.use_film = use_film
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit*cnn_multipler[0], (7, 7), padding=3),
            nn.BatchNorm2d(cnn_unit*cnn_multipler[0]),
            nn.ReLU(),
            nn.MaxPool2d((1, 4)),

            nn.Dropout(0.25),

            nn.Conv2d(cnn_unit*cnn_multipler[0], cnn_unit*cnn_multipler[1], (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit*cnn_multipler[1]),
            nn.ReLU(),
            nn.Conv2d(cnn_unit*cnn_multipler[1], cnn_unit*cnn_multipler[2], (3, 1), padding=(1,0)),
            nn.BatchNorm2d(cnn_unit*cnn_multipler[2]),
            nn.ReLU(),
        )
        f_size = 88 
        self.large_conv_l1 = nn.Conv2d(cnn_unit*cnn_multipler[2], hidden_per_pitch, (1, 88), padding=0, stride=(1,2))
        self.large_conv_l2 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, 1), padding=0)
        self.large_conv_l3 = nn.Conv2d(hidden_per_pitch, hidden_per_pitch, (1, 1), padding=0)
        if use_film:
            self.film_1 = FilmLayer(88, hidden_per_pitch, hidden=16)
            self.film_2 = FilmLayer(88, hidden_per_pitch, hidden=16)
            self.film_3 = FilmLayer(88, hidden_per_pitch, hidden=16)
       
        self.fc_0 = nn.Sequential(
            nn.Conv2d(cnn_unit*cnn_multipler[2], 4, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1, padding=0),
            nn.ReLU(),
        )
        self.fc_1 = nn.Conv1d(790//4, hidden_per_pitch*88, 1, padding=0)
        self.fc_2 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)
        self.fc_3 = nn.Conv1d(hidden_per_pitch*88, hidden_per_pitch*88, 1, padding=0, groups=88)

        self.window_cnn = nn.Sequential(
            nn.ZeroPad2d((0, 0, self.win_bw, self.win_fw)),
            nn.Conv2d(hidden_per_pitch*2, hidden_per_pitch*2, (self.win_bw + self.win_fw + 1, 1))
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        batch_size = x.shape[0]
        cnn_out = self.cnn(x) # (B x H x L x n_mels/4)
        cnn_out = cnn_out.contiguous()
        # print(f'cnn_out:{cnn_out.shape}')
        if self.use_film:
            
            x = self.large_conv_l1(F.pad(cnn_out, (20,46)))
            x = x[:,:,:,:88]
            x = F.relu(self.film_1(x))
            x = self.large_conv_l2(x)
            x = F.relu(self.film_2(x))
            x = self.large_conv_l3(x)
            x = F.relu(self.film_3(x))
        else:
            x = self.large_conv_l1(cnn_out)[:,:,:,:88]
            x = F.relu(x)
            x = self.large_conv_l2(x)
            x = F.relu(x)
            x = self.large_conv_l3(x)
            x = F.relu(x)
        x_pitchwise = self.fc_0(cnn_out)  # B x 1 x L x n_mels/4
        x_pitchwise = x_pitchwise.transpose(1, 2).flatten(-2).transpose(1,2)  # (B x n_mels/4 x L)

        x_pitchwise = F.relu(self.fc_1(x_pitchwise)) # B x H*88 x L
        res = self.fc_2(x_pitchwise.contiguous())
        res = self.fc_3(F.relu(res.contiguous()))
        x_pitchwise = x_pitchwise + res
        x_pitchwise = x_pitchwise.reshape(batch_size, self.hidden_per_pitch, 88, -1).permute(0, 1, 3, 2)
        # B, H, L, 88

        x = th.cat((x_pitchwise, x), 1)
        x = self.window_cnn(x)  # B x H x L x 88
        x = x.transpose(1, 2)
        # x = x.flatten(-2)

        return x  # B x L x H x 88