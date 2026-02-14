import torch
import torch.nn as nn
import math
import torch.nn.init as init
#Implementation of the ASTR-Net architecture 
def get_sinusoidal_absolute_positional_encoding(num_positions: int, embedding_dim: int) -> torch.Tensor:
    pe = torch.zeros(num_positions, embedding_dim)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    if embedding_dim % 2 != 0:
        valid_div_term_len = (embedding_dim + 1) // 2
        if div_term.shape[0] >= valid_div_term_len:
             pe[:, 1::2] = torch.cos(position * div_term[:valid_div_term_len])
        else:
             pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class EnhancedTemporalProjection(nn.Module):
    def __init__(self,
                 num_brain_regions: int,
                 conv_kernel_size: int = 21,
                 num_conv_layers: int = 2,
                 activation_fn: str = 'ReLU',
                 init_std: float = 0.01,
                 init_temporal_conv_to_zero: bool = False):
        super().__init__()
        self.num_brain_regions = num_brain_regions

        layers = []
        padding = (conv_kernel_size - 1) // 2
        current_channels = num_brain_regions 

        for i in range(num_conv_layers):
            conv_layer = nn.Conv1d(
                in_channels=current_channels, 
                out_channels=num_brain_regions, 
                kernel_size=conv_kernel_size,
                padding=padding,
                groups=994,
                bias=True 
            )
            layers.append(conv_layer)
            
            if i < num_conv_layers - 1:
                try:
                    activation = getattr(nn, activation_fn)()
                    layers.append(activation)
                except AttributeError:
                    print(f"Warning: activation function '{activation_fn}' not found, using ReLU instead.")
                    layers.append(nn.ReLU())

        self.correction_network = nn.Sequential(*layers)
        
        if init_temporal_conv_to_zero:
            print("Initializing EnhancedTemporalProjection Conv1d weights and bias to ZERO.")
            for layer in self.correction_network:
                if isinstance(layer, nn.Conv1d):
                    init.constant_(layer.weight, 0.0)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0.0)
        else:
            print(f"Initializing EnhancedTemporalProjection Conv1d weights with N(0, {init_std}) and bias to zero.")
            for layer in self.correction_network:
                if isinstance(layer, nn.Conv1d):
                    init.normal_(layer.weight, mean=0.0, std=init_std)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0.0)

    def forward(self, s_loc_raw: torch.Tensor) -> torch.Tensor:
        s_loc_raw_permuted = s_loc_raw.permute(0, 2, 1)
        s_waveform_correction_permuted = self.correction_network(s_loc_raw_permuted)
        s_waveform_correction = s_waveform_correction_permuted.permute(0, 2, 1)
        final_output = s_loc_raw + s_waveform_correction
        return final_output

class SpatialAttentionConvAggregator(nn.Module):
    def __init__(
        self,
        num_hidden: int,
        expected_num_sensors: int,
        qkv_dim: int = 128,
        num_heads: int = 4,
        activation: str = 'GELU',
        dropout_rate: float = 0.1
    ):
        super().__init__()

        if qkv_dim % num_heads != 0:
            raise ValueError(f"qkv_dim ({qkv_dim}) must be divisible by num_heads ({num_heads}).")

        self.num_sensor = expected_num_sensors
        self.num_hidden = num_hidden
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads

        abs_pos_encoding = get_sinusoidal_absolute_positional_encoding(self.num_sensor, self.qkv_dim)
        self.register_buffer("absolute_positional_encoding", abs_pos_encoding)

        self.q_conv = nn.Conv1d(in_channels=1, out_channels=qkv_dim, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(in_channels=1, out_channels=qkv_dim, kernel_size=1, bias=False)
        self.v_conv = nn.Conv1d(in_channels=1, out_channels=qkv_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Linear(qkv_dim, qkv_dim, bias=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.depthwise_conv = nn.Conv1d(
            in_channels=qkv_dim,
            out_channels=qkv_dim,
            kernel_size=self.num_sensor,
            groups=qkv_dim,
            bias=False
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=qkv_dim,
            out_channels=num_hidden,
            kernel_size=1,
            bias=True
        )

        try:
            self.activation_fn_internal = getattr(nn, activation)()
        except AttributeError:
            print(f"Warning: activation function '{activation}' not found, using ReLU instead.")
            self.activation_fn_internal = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> dict:
        N, T, C = x.shape
        if C != self.num_sensor:
            raise ValueError(f"Input tensor last dimension (C={C}) does not match expected_num_sensors ({self.num_sensor}).")

        x_flat_time = x.reshape(N * T, C)
        x_flat_time_embed = x_flat_time.unsqueeze(-1)
        x_permuted = x_flat_time_embed.permute(0, 2, 1)

        q_proj = self.q_conv(x_permuted)
        k_proj = self.k_conv(x_permuted)
        v_proj = self.v_conv(x_permuted)

        pe_broadcastable = self.absolute_positional_encoding.transpose(0, 1).unsqueeze(0)
        q_proj_with_ape = q_proj + pe_broadcastable
        k_proj_with_ape = k_proj + pe_broadcastable

        batch_size_time = q_proj_with_ape.size(0)
        h = self.num_heads
        d_h = self.head_dim

        q = q_proj_with_ape.view(batch_size_time, h, d_h, C).transpose(2, 3)
        k = k_proj_with_ape.view(batch_size_time, h, d_h, C).transpose(2, 3)
        v = v_proj.view(batch_size_time, h, d_h, C).transpose(2, 3)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_h)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output_heads = torch.matmul(attn_weights, v)

        attn_output_concat = (
            attn_output_heads
            .transpose(1, 2)
            .contiguous()
            .view(batch_size_time, C, self.qkv_dim)
        )
        attn_output = self.out_proj(attn_output_concat)
        attn_output_dropout = self.dropout1(attn_output)

        v_orig_reshaped = v_proj.permute(0, 2, 1)
        attn_res = v_orig_reshaped + attn_output_dropout

        attn_output_perm = attn_res.permute(0, 2, 1)

        depthwise_out = self.depthwise_conv(attn_output_perm)
        activated_depthwise = self.activation_fn_internal(depthwise_out)

        aggregated = self.pointwise_conv(activated_depthwise)
        aggregated_squeezed = aggregated.squeeze(-1)

        activated_output = self.activation_fn_internal(aggregated_squeezed)
        final_output_flat = self.dropout2(activated_output)

        final_output = final_output_flat.view(N, T, self.num_hidden)
        return {'value_activation': final_output}

class TemporalModuleWithProjection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        initial_conv_dim: int,
        initial_conv_kernel_size: int,
        gru_hidden_size_per_direction: int,
        num_gru_layers: int,
        gru_bidirectional: bool,
        gru_dropout_rate: float,
        num_brain_regions: int,
        enh_proj_conv_kernel_size: int,
        enh_proj_num_layers: int,
        enh_proj_activation: str,
        enh_proj_init_std: float,
        enh_proj_init_temporal_conv_to_zero: bool
    ):
        super().__init__()

        padding = (initial_conv_kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=initial_conv_dim,
            kernel_size=initial_conv_kernel_size,
            padding=padding,
            bias=False
        )
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(gru_dropout_rate)

        self.gru = nn.GRU(
            input_size=initial_conv_dim,
            hidden_size=gru_hidden_size_per_direction,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=gru_dropout_rate if num_gru_layers > 1 else 0.0,
            bidirectional=gru_bidirectional
        )

        gru_actual_output_dim = gru_hidden_size_per_direction * (2 if gru_bidirectional else 1)
        if gru_actual_output_dim != num_brain_regions:
            raise ValueError(
                f"GRU's effective output dimension ({gru_actual_output_dim}) must be equal to "
                f"num_brain_regions ({num_brain_regions}). Check 'gru_hidden_size_per_direction'."
            )

        self.projection_head = EnhancedTemporalProjection(
            num_brain_regions=num_brain_regions,
            conv_kernel_size=enh_proj_conv_kernel_size,
            num_conv_layers=enh_proj_num_layers,
            activation_fn=enh_proj_activation,
            init_std=enh_proj_init_std,
            init_temporal_conv_to_zero=enh_proj_init_temporal_conv_to_zero
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1)
        conv_out = self.conv1(x_perm)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_dropout(conv_out)
        conv_out_for_gru = conv_out.permute(0, 2, 1)
        gru_out, _ = self.gru(conv_out_for_gru)
        final_output = self.projection_head(gru_out)
        return final_output

class EEGSourceLocalizationModel(nn.Module):
    def __init__(
        self,
        num_sensors: int = 75,
        spatial_output_dim: int = 512,
        qkv_dim: int = 256,
        num_heads: int = 8,
        spatial_activation: str = 'GELU',
        spatial_dropout: float = 0.1,
        temporal_initial_conv_dim: int = 512,
        temporal_initial_conv_kernel_size: int = 21,
        temporal_gru_hidden_size_per_direction: int = 994,
        temporal_gru_layers: int = 4,
        temporal_gru_dropout: float = 0.3,
        temporal_gru_bidirectional: bool = False,
        num_brain_regions: int = 994,
        enh_proj_conv_kernel_size: int = 7,
        enh_proj_num_layers: int = 3,
        enh_proj_activation: str = 'ReLU',
        enh_proj_init_std: float = 0.1,
        enh_proj_init_temporal_conv_to_zero: bool = False
    ):
        super().__init__()

        self.spatial_module = SpatialAttentionConvAggregator(
            num_hidden=spatial_output_dim,
            expected_num_sensors=num_sensors,
            qkv_dim=qkv_dim,
            num_heads=num_heads,
            activation=spatial_activation,
            dropout_rate=spatial_dropout
        )

        expected_gru_output_channels = temporal_gru_hidden_size_per_direction * (2 if temporal_gru_bidirectional else 1)
        if expected_gru_output_channels != num_brain_regions:
            raise ValueError(
                f"Configuration Mismatch for GRU output! Effective GRU output dim: {expected_gru_output_channels}, "
                f"num_brain_regions: {num_brain_regions}. Adjust 'temporal_gru_hidden_size_per_direction'."
            )
        if temporal_gru_bidirectional and num_brain_regions % 2 != 0:
            raise ValueError(
                f"If GRU is bidirectional, 'num_brain_regions' ({num_brain_regions}) must be even."
            )

        self.temporal_module = TemporalModuleWithProjection(
            input_dim=spatial_output_dim,
            initial_conv_dim=temporal_initial_conv_dim,
            initial_conv_kernel_size=temporal_initial_conv_kernel_size,
            gru_hidden_size_per_direction=temporal_gru_hidden_size_per_direction,
            num_gru_layers=temporal_gru_layers,
            gru_bidirectional=temporal_gru_bidirectional,
            gru_dropout_rate=temporal_gru_dropout,
            num_brain_regions=num_brain_regions,
            enh_proj_conv_kernel_size=enh_proj_conv_kernel_size,
            enh_proj_num_layers=enh_proj_num_layers,
            enh_proj_activation=enh_proj_activation,
            enh_proj_init_std=enh_proj_init_std,
            enh_proj_init_temporal_conv_to_zero=enh_proj_init_temporal_conv_to_zero
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_output_dict = self.spatial_module(x)
        spatial_features = spatial_output_dict['value_activation']
        final_output = self.temporal_module(spatial_features)
        return final_output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
