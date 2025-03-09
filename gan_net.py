import torch
import torch.nn as nn

# 初始化权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0)

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# SE
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width)
        out = self.gamma * out + x
        return out
    
#平滑光谱
class SmoothingLayer(nn.Module):
    def __init__(self, kernel_size=5):
        super(SmoothingLayer, self).__init__()
        self.smoothing = nn.Sequential(
            nn.ReflectionPad1d(kernel_size // 2),
            nn.AvgPool1d(kernel_size=kernel_size, stride=1),
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)  # 微调平滑后的结果
        )
        
    def forward(self, x):
        return self.smoothing(x)
    
#生成器
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.ConvTranspose1d(nz, 1024, kernel_size=16, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        
        # 增加残差块数量，提升特征提取能力
        self.res_blocks = nn.Sequential(
            ResidualBlock(1024),
            ResidualBlock(1024),
            ResidualBlock(1024),
            SelfAttention(1024)  # 添加自注意力层
        )
        
        # 使用更小的kernel_size和更多的层来实现更精细的上采样
        self.upsampling = nn.Sequential(
            # 1024 -> 512, 32
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            ResidualBlock(512),
            
            # 512 -> 256, 64
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            ResidualBlock(256),
            
            # 256 -> 128, 128
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            ResidualBlock(128),
            
            # 128 -> 64, 256
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            ResidualBlock(64),
            
            # 64 -> 32, 512
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )
        
        # 最终输出层，包含平滑处理
        self.final = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=1, bias=False),
            nn.Tanh(),
            SmoothingLayer(kernel_size=5)  # 添加平滑层
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.upsampling(x)
        x = self.final(x)
        return x
    
#判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: [batch_size, 1, 512]
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # -> [b, 64, 256]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # -> [b, 128, 128]
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            ResidualBlock(128),
            
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # -> [b, 256, 64]
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # -> [b, 512, 32]
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(512, 1, kernel_size=32, stride=1, padding=0, bias=False),  # -> [b, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)