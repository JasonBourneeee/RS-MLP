import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Data_RS import RamanDataset
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        nn.init.kaiming_normal_(self.conv[0].weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        residual = x
        # 转置仅用于卷积，不影响 LayerNorm
        out = self.conv1(F.relu(self.norm1(x.transpose(1, 2))).transpose(1, 2))
        out = self.conv2(F.relu(self.norm2(out.transpose(1, 2))).transpose(1, 2))
        return F.relu(out + residual)
    
class MultiHeadAttention(nn.Module):

    def __init__(self, feature_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.all_head_dim = self.num_heads * self.head_dim
        self.query = nn.Linear(feature_dim, self.all_head_dim)
        self.key = nn.Linear(feature_dim, self.all_head_dim)
        self.value = nn.Linear(feature_dim, self.all_head_dim)
        self.out = nn.Linear(self.all_head_dim, feature_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        

        self.attention_maps = {
            'position_intensities': [],
            'peak_matches': [],     
            'input_features': [],       
            'output_features': []       
        }
        
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)
        
    def forward(self, query_input, key_value_input=None):
        if key_value_input is None:
            key_value_input = query_input
            

        self.attention_maps['input_features'].append({
            'query': query_input.detach(),
            'key': key_value_input.detach()
        })
            
        query = self.transpose_for_scores(self.query(query_input))
        key = self.transpose_for_scores(self.key(key_value_input))
        value = self.transpose_for_scores(self.value(key_value_input))
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_probs = self.softmax(attention_scores)
        

        attention_info = {
            'weights': attention_probs.detach(),  # [batch_size, num_heads, seq_len, seq_len]
            'query_shape': query.shape,
            'key_shape': key.shape,
            'scores': attention_scores.detach()
        }

        if query_input.shape == key_value_input.shape:
            self.attention_maps['position_intensities'].append(attention_info)
        else:
            self.attention_maps['peak_matches'].append(attention_info)
        
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), -1, self.all_head_dim)
        output = self.out(context_layer)
        

        self.attention_maps['output_features'].append(output.detach())
        
        return output
    
    def get_attention_maps(self):

        return self.attention_maps
    
    def clear_attention_maps(self):

        self.attention_maps = {
            'position_intensities': [],
            'peak_matches': [],
            'input_features': [],
            'output_features': []
        }
    
class MatchingMLPMixer(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        

        self.local_feature_net = nn.Sequential(
            nn.Conv1d(1, hidden_dim//4, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//4, hidden_dim//4, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//4, hidden_dim//4, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(inplace=True)
        )
        

        self.feature_encoder = nn.Sequential(
            nn.Linear(hidden_dim//4 + 2, hidden_dim//2),  # +2 for position and intensity
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, input_dim),
            nn.LayerNorm(input_dim)
        )
        

        self.peak_shape_net = nn.Sequential(
            nn.Conv1d(1, hidden_dim//4, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        

        self.token_mixing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        
        self.channel_mixing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )
        

        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

        self.peak_attention = MultiHeadAttention(
            feature_dim=input_dim,
            num_heads=4,
            dropout_rate=0.1
        )
        
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, pure_data, pure_intensities, mixture_features):
        batch_size = mixture_features.size(0)

        pure_data_expanded = pure_data.unsqueeze(1)
        local_features = self.local_feature_net(pure_data_expanded)
        local_features = local_features.transpose(1, 2)

        shape_features = self.peak_shape_net(pure_data_expanded)
        shape_features = shape_features.squeeze(-1)

        combined_features = torch.cat([
            local_features,
            pure_data.unsqueeze(-1),
            pure_intensities.unsqueeze(-1)
        ], dim=-1)
        
        feature_embeddings = self.feature_encoder(combined_features)

        token_mixed = self.token_mixing(mixture_features)
        mixture_features = token_mixed + mixture_features
        

        channel_mixed = self.channel_mixing(mixture_features)
        mixture_features = channel_mixed + mixture_features
        
        matched_features = self.peak_attention(
            query_input=mixture_features,
            key_value_input=feature_embeddings
        )

        feature_importance = self.softmax(self.feature_weights)
        weighted_features = matched_features * feature_importance
        
        fused_features = self.fusion(torch.cat([mixture_features, weighted_features], dim=-1))
        final_features = fused_features + mixture_features
        
        return final_features
            
    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class ImprovedRamanModel(nn.Module):
    def __init__(self, dataset, input_length=128, concentration_threshold=0.005):
        super(ImprovedRamanModel, self).__init__()
        
        self.peak_features = dataset.peak_features
        self.input_length = input_length
        self.concentration_threshold = concentration_threshold


        self.conv_block1 = ConvBlock(1, 32, kernel_size=7, padding=3)
        self.res_block1 = ResBlock(32)
        self.conv_block2 = ConvBlock(32, 64, kernel_size=5, padding=2)
        self.res_block2 = ResBlock(64)
        self.conv_block3 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.res_block3 = ResBlock(128)

        self.proj_32_to_128 = nn.Conv1d(32, 128, 1)
        self.proj_64_to_128 = nn.Conv1d(64, 128, 1)
        
        self.scale_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))

        self.position_intensity_attention = MultiHeadAttention(
            feature_dim=128,
            num_heads=4,
            dropout_rate=0.1
        )

        self.mlp_mixer = MatchingMLPMixer(
            input_dim=128,
            hidden_dim=256
        )
        
        self.predict_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(self.peak_features))
        )
        
        self.uncertainty_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, len(self.peak_features)),
            nn.Softplus()
        )
        nn.init.zeros_(self.uncertainty_fc[-2].weight)
        nn.init.constant_(self.uncertainty_fc[-2].bias, -1.0)

        self.component_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(self.peak_features)),
            nn.Sigmoid()
        )

    def forward(self, x):

        x1 = self.conv_block1(x)
        x1 = self.res_block1(x1)
        
        x2 = self.conv_block2(x1)
        x2 = self.res_block2(x2)
        
        x3 = self.conv_block3(x2)
        x3 = self.res_block3(x3)
        

        x1_proj = self.proj_32_to_128(x1)
        x2_proj = self.proj_64_to_128(x2)
        
        x1_resized = F.interpolate(x1_proj, size=x3.shape[2])
        x2_resized = F.interpolate(x2_proj, size=x3.shape[2])

        multi_scale_weights = F.softmax(self.scale_weights, dim=0)
        multi_scale_features = (
            x1_resized * multi_scale_weights[0] +
            x2_resized * multi_scale_weights[1] +
            x3 * multi_scale_weights[2]
        )
        
        # [B, L, C]
        multi_scale_features = multi_scale_features.transpose(1, 2)
        
        enhanced_features = self.position_intensity_attention(multi_scale_features)
        
        pure_peaks_list = []
        pure_heights_list = []
        for peak_info in self.peak_features.values():
            pos = torch.tensor(peak_info['positions'], dtype=torch.float32)
            height = torch.tensor(peak_info['heights'], dtype=torch.float32)
            height = (height / height.max()) * 100
            pure_peaks_list.append(pos)
            pure_heights_list.append(height)
        
        pure_peaks = torch.stack(pure_peaks_list).to(x.device)
        pure_heights = torch.stack(pure_heights_list).to(x.device)
        
        batch_size = x.size(0)
        pure_peaks = pure_peaks.unsqueeze(0).expand(batch_size, -1, -1)
        pure_heights = pure_heights.unsqueeze(0).expand(batch_size, -1, -1)
        
        matched_features_list = []
        for i in range(pure_peaks.size(1)):
            matched = self.mlp_mixer(
                pure_peaks[:, i],
                pure_heights[:, i],
                enhanced_features
            )
            matched_features_list.append(matched)
        
        matched_features = torch.stack(matched_features_list, dim=1)
        
        fused_features = matched_features.mean(dim=1)
        
        pooled_features = fused_features.mean(dim=1)
        raw_concentrations = self.predict_fc(pooled_features)
        concentrations = F.softmax(raw_concentrations, dim=-1)
        uncertainties = self.uncertainty_fc(pooled_features)
        component_presence = self.component_classifier(pooled_features)
        
        mask = (concentrations > self.concentration_threshold) & (component_presence > 0.5)
        final_concentrations = concentrations * mask.float()
        final_concentrations = final_concentrations / (final_concentrations.sum(dim=-1, keepdim=True) + 1e-6)
        
        return final_concentrations, uncertainties, component_presence
    
    def predict_with_threshold(self, x, uncertainty_threshold=0.1, confidence_level=0.95):

        self.eval()
        with torch.no_grad():
            concentrations, uncertainties, _ = self(x)
            
            z_score = 1.96  
            confidence_intervals = z_score * torch.sqrt(uncertainties)
            
            relative_uncertainty = uncertainties / (concentrations + 1e-6)
            uncertain_mask = relative_uncertainty < uncertainty_threshold
            filtered_concentrations = concentrations * uncertain_mask.float()
            
            total_conc = filtered_concentrations.sum(dim=-1, keepdim=True) + 1e-6
            filtered_concentrations = filtered_concentrations / total_conc
            
            return (filtered_concentrations, 
                    uncertainties,
                    {
                        'confidence_intervals': confidence_intervals,
                        'relative_uncertainty': relative_uncertainty
                    })

    def calculate_attention_weights(self, x):

        self.eval()
        with torch.no_grad():
            x1 = self.conv_block1(x)
            x1 = self.res_block1(x1)
            x2 = self.conv_block2(x1)
            x2 = self.res_block2(x2)
            x3 = self.conv_block3(x2)
            x3 = self.res_block3(x3)
            
            x1_proj = self.proj_32_to_128(x1)
            x2_proj = self.proj_64_to_128(x2)
            
            x1_resized = F.interpolate(x1_proj, size=x3.shape[2])
            x2_resized = F.interpolate(x2_proj, size=x3.shape[2])
            
            multi_scale_weights = F.softmax(self.scale_weights, dim=0)
            multi_scale_features = (
                x1_resized * multi_scale_weights[0] +
                x2_resized * multi_scale_weights[1] +
                x3 * multi_scale_weights[2]
            )
            
            features = multi_scale_features.transpose(1, 2)
            
            enhanced_features = self.position_intensity_attention(features)
            
            batch_size = x.size(0)
            pure_peaks_list = []
            pure_heights_list = []
            
            for peak_info in self.peak_features.values():
                pos = torch.tensor(peak_info['positions'], dtype=torch.float32).to(x.device)
                height = torch.tensor(peak_info['heights'], dtype=torch.float32).to(x.device)
                height = (height / height.max()) * 100
                pure_peaks_list.append(pos)
                pure_heights_list.append(height)
            
            pure_peaks = torch.stack(pure_peaks_list).unsqueeze(0).expand(batch_size, -1, -1)
            pure_heights = torch.stack(pure_heights_list).unsqueeze(0).expand(batch_size, -1, -1)
            
            component_features = []
            for i in range(pure_peaks.size(1)):
                matched = self.mlp_mixer(
                    pure_peaks[:, i],
                    pure_heights[:, i],
                    enhanced_features
                )
                component_features.append(matched)
            
            return {
                'multi_scale_features': multi_scale_features.detach(),
                'enhanced_features': enhanced_features.detach(),
                'component_features': torch.stack(component_features, dim=1).detach(),
                'attention_weights': self.position_intensity_attention.last_attention_weights,
                'pure_peaks': pure_peaks.detach(),
                'pure_heights': pure_heights.detach(),
                'conv_features': {
                    'x1': x1.detach(),
                    'x2': x2.detach(),
                    'x3': x3.detach()
                }
            }

