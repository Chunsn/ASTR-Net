
import math

import torch
import torch.nn as nn
import torch.nn.init as init


def get_sinusoidal_absolute_positional_encoding(
    num_positions: int,
    embedding_dim: int,
) -> torch.Tensor:
    pe = torch.zeros(num_positions, embedding_dim)
    position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float()
        * (-math.log(10000.0) / embedding_dim)
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
    def __init__(
        self,
        num_brain_regions: int,
        conv_kernel_size: int = 21,
        num_conv_layers: int = 2,
        activation_fn: str = "ReLU",
        init_std: float = 0.01,
        init_temporal_conv_to_zero: bool = False,
    ):
        super().__init__()
        self.num_brain_regions = num_brain_regions

        layers = []
        padding = (conv_kernel_size - 1) // 2
        for index in range(num_conv_layers):
            conv_layer = nn.Conv1d(
                in_channels=num_brain_regions,
                out_channels=num_brain_regions,
                kernel_size=conv_kernel_size,
                padding=padding,
                groups=num_brain_regions,
                bias=True,
            )
            layers.append(conv_layer)
            if index < num_conv_layers - 1:
                try:
                    layers.append(getattr(nn, activation_fn)())
                except AttributeError:
                    layers.append(nn.ReLU())

        self.correction_network = nn.Sequential(*layers)

        if init_temporal_conv_to_zero:
            for layer in self.correction_network:
                if isinstance(layer, nn.Conv1d):
                    init.constant_(layer.weight, 0.0)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0.0)
        else:
            for layer in self.correction_network:
                if isinstance(layer, nn.Conv1d):
                    init.normal_(layer.weight, mean=0.0, std=init_std)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0.0)

    def forward(self, s_loc_raw: torch.Tensor) -> torch.Tensor:
        source_channels_first = s_loc_raw.permute(0, 2, 1)
        correction_channels_first = self.correction_network(source_channels_first)
        correction = correction_channels_first.permute(0, 2, 1)
        return s_loc_raw + correction


class SpatialAttentionConvAggregator(nn.Module):
    def __init__(
        self,
        num_hidden: int,
        expected_num_sensors: int,
        qkv_dim: int = 128,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if qkv_dim % num_heads != 0:
            raise ValueError(
                f"qkv_dim ({qkv_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_sensor = expected_num_sensors
        self.num_hidden = num_hidden
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads

        positional_encoding = get_sinusoidal_absolute_positional_encoding(
            self.num_sensor,
            self.qkv_dim,
        )
        self.register_buffer("absolute_positional_encoding", positional_encoding)

        self.q_conv = nn.Conv1d(1, qkv_dim, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(1, qkv_dim, kernel_size=1, bias=False)
        self.v_conv = nn.Conv1d(1, qkv_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Linear(qkv_dim, qkv_dim, bias=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.depthwise_conv = nn.Conv1d(
            in_channels=qkv_dim,
            out_channels=qkv_dim,
            kernel_size=self.num_sensor,
            groups=qkv_dim,
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=qkv_dim,
            out_channels=num_hidden,
            kernel_size=1,
            bias=True,
        )

        try:
            self.activation_fn_internal = getattr(nn, activation)()
        except AttributeError:
            self.activation_fn_internal = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> dict:
        batch_size, time_points, sensors = x.shape
        if sensors != self.num_sensor:
            raise ValueError(
                f"Input sensor dimension ({sensors}) does not match "
                f"expected_num_sensors ({self.num_sensor})."
            )

        x_flat_time = x.reshape(batch_size * time_points, sensors)
        x_permuted = x_flat_time.unsqueeze(-1).permute(0, 2, 1)

        q_proj = self.q_conv(x_permuted)
        k_proj = self.k_conv(x_permuted)
        v_proj = self.v_conv(x_permuted)

        positional_encoding = (
            self.absolute_positional_encoding.transpose(0, 1).unsqueeze(0)
        )
        q_proj = q_proj + positional_encoding
        k_proj = k_proj + positional_encoding

        batch_time = q_proj.size(0)
        heads = self.num_heads
        head_dim = self.head_dim

        q = q_proj.view(batch_time, heads, head_dim, sensors).transpose(2, 3)
        k = k_proj.view(batch_time, heads, head_dim, sensors).transpose(2, 3)
        v = v_proj.view(batch_time, heads, head_dim, sensors).transpose(2, 3)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_heads = torch.matmul(attention_weights, v)
        attention_concat = (
            attention_heads.transpose(1, 2)
            .contiguous()
            .view(batch_time, sensors, self.qkv_dim)
        )
        attention_output = self.dropout1(self.out_proj(attention_concat))

        value_residual = v_proj.permute(0, 2, 1) + attention_output
        depthwise_input = value_residual.permute(0, 2, 1)
        depthwise_output = self.depthwise_conv(depthwise_input)
        depthwise_output = self.activation_fn_internal(depthwise_output)
        aggregated = self.pointwise_conv(depthwise_output).squeeze(-1)
        aggregated = self.activation_fn_internal(aggregated)
        aggregated = self.dropout2(aggregated)

        spatial_features = aggregated.view(batch_size, time_points, self.num_hidden)
        return {"value_activation": spatial_features}


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
        enh_proj_init_temporal_conv_to_zero: bool,
    ):
        super().__init__()

        padding = (initial_conv_kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=initial_conv_dim,
            kernel_size=initial_conv_kernel_size,
            padding=padding,
            bias=False,
        )
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(gru_dropout_rate)

        self.gru = nn.GRU(
            input_size=initial_conv_dim,
            hidden_size=gru_hidden_size_per_direction,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=gru_dropout_rate if num_gru_layers > 1 else 0.0,
            bidirectional=gru_bidirectional,
        )

        gru_output_dim = gru_hidden_size_per_direction * (
            2 if gru_bidirectional else 1
        )
        if gru_output_dim != num_brain_regions:
            raise ValueError(
                f"GRU output dimension ({gru_output_dim}) must equal "
                f"num_brain_regions ({num_brain_regions})."
            )

        self.projection_head = EnhancedTemporalProjection(
            num_brain_regions=num_brain_regions,
            conv_kernel_size=enh_proj_conv_kernel_size,
            num_conv_layers=enh_proj_num_layers,
            activation_fn=enh_proj_activation,
            init_std=enh_proj_init_std,
            init_temporal_conv_to_zero=enh_proj_init_temporal_conv_to_zero,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        convolved = self.conv1(x.permute(0, 2, 1))
        convolved = self.conv_dropout(self.relu(convolved))
        gru_input = convolved.permute(0, 2, 1)
        coarse_source, _ = self.gru(gru_input)
        return self.projection_head(coarse_source)


class EEGSourceLocalizationModel(nn.Module):
    def __init__(
        self,
        num_sensors: int = 75,
        spatial_output_dim: int = 512,
        qkv_dim: int = 256,
        num_heads: int = 8,
        spatial_activation: str = "GELU",
        spatial_dropout: float = 0.1,
        temporal_initial_conv_dim: int = 256,
        temporal_initial_conv_kernel_size: int = 21,
        temporal_gru_hidden_size_per_direction: int = 994,
        temporal_gru_layers: int = 3,
        temporal_gru_dropout: float = 0.3,
        temporal_gru_bidirectional: bool = False,
        num_brain_regions: int = 994,
        enh_proj_conv_kernel_size: int = 7,
        enh_proj_num_layers: int = 3,
        enh_proj_activation: str = "ReLU",
        enh_proj_init_std: float = 0.1,
        enh_proj_init_temporal_conv_to_zero: bool = False,
    ):
        super().__init__()

        self.spatial_module = SpatialAttentionConvAggregator(
            num_hidden=spatial_output_dim,
            expected_num_sensors=num_sensors,
            qkv_dim=qkv_dim,
            num_heads=num_heads,
            activation=spatial_activation,
            dropout_rate=spatial_dropout,
        )

        expected_gru_output_channels = temporal_gru_hidden_size_per_direction * (
            2 if temporal_gru_bidirectional else 1
        )
        if expected_gru_output_channels != num_brain_regions:
            raise ValueError(
                f"GRU output dimension ({expected_gru_output_channels}) must equal "
                f"num_brain_regions ({num_brain_regions})."
            )
        if temporal_gru_bidirectional and num_brain_regions % 2 != 0:
            raise ValueError(
                "num_brain_regions must be even when a bidirectional GRU is used."
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
            enh_proj_init_temporal_conv_to_zero=enh_proj_init_temporal_conv_to_zero,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_module(x)["value_activation"]
        return self.temporal_module(spatial_features)

    def count_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
