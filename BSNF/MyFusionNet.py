# coding:utf-8
import torch.nn as nn
from Compute_Kernels import *
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置按需分配内存（而非预分配所有内存）
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


class ConvBnRelu2d(nn.Module):
    # convolution
    # batch normalization
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.conv(x))

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        # x = torch.cat((x, self.conv3(x)), dim=1)
        return x

class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

class drf_block(nn.Module):
    def __init__(self,in_channels,out_channels,G=4):
        super(drf_block, self).__init__()
        self.features = out_channels
        self.convs_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=G,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.convs_layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=G,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
        self.convs_layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, groups=G,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        # self.dense =DenseBlock(in_channels)
        # self.convdown=Conv1(3*in_channels,out_channels)
        # self.sobelconv=Sobelxy(in_channels)
        # self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):

        batch_size = x.shape[0]
        feats_1 = self.convs_layer1(x)
        feats_2 = self.convs_layer2(feats_1)
        feats_3 = self.convs_layer3(feats_2)
        feats = torch.cat((feats_1,feats_2,feats_3), dim=1)
        feats = feats.view(batch_size, 3, self.features, feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)
        return feats_U

class MyFusionNet(nn.Module):
    def __init__(self, output):
        super(MyFusionNet, self).__init__()
        vis_ch = [16,16,16,16]
        inf_ch = [16,16,16,16]
        output=1
        self.vis_conv=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1 = drf_block(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = drf_block(vis_ch[0]+vis_ch[1], vis_ch[2])
        self.vis_rgbd3 = drf_block(vis_ch[0]+vis_ch[1]+vis_ch[2], vis_ch[3])

        self.inf_conv=ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = drf_block(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = drf_block(inf_ch[0]+inf_ch[1], inf_ch[2])
        self.inf_rgbd3 = drf_block(inf_ch[0]+inf_ch[1]+inf_ch[2], inf_ch[3])

        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[0]+vis_ch[1]+vis_ch[2]+vis_ch[3],vis_ch[0]+vis_ch[1]+vis_ch[2])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[0]+vis_ch[1]+vis_ch[2],vis_ch[0]+vis_ch[1])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0]+vis_ch[1], vis_ch[0])
        self.decode1 = ConvBnTanh2d(vis_ch[0], output)

        self.vis_weight1 = torch.nn.Conv2d(vis_ch[0] + vis_ch[1], 1, kernel_size=1, stride=1, padding=0,
                                           bias=True)
        self.vis_weight2 = torch.nn.Conv2d(vis_ch[0] + vis_ch[1]+vis_ch[2] , 1, kernel_size=1, stride=1, padding=0,
                                           bias=True)
        self.vis_weight3 = torch.nn.Conv2d(vis_ch[0]+vis_ch[1]+vis_ch[2]+vis_ch[3],1, kernel_size=1,stride=1, padding=0,
                                          bias=True)

        self.inf_weight1 = torch.nn.Conv2d(inf_ch[0], 1, kernel_size=1, stride=1, padding=0,
                                           bias=True)
        self.inf_weight2 = torch.nn.Conv2d(inf_ch[0] + inf_ch[1] , 1, kernel_size=1, stride=1, padding=0,
                                           bias=True)
        self.inf_weight3 = torch.nn.Conv2d(inf_ch[0]+inf_ch[1]+inf_ch[2], 1, kernel_size=1, stride=1, padding=0,
                                          bias=True)

    def forward(self, image_vis,image_ir):
        # split data into RGB and INF
        x_vis_origin = image_vis[:,:1]
        x_inf_origin = image_ir
        # encode
        x_vis_p = self.vis_conv(x_vis_origin)
        x_vis_p1 = torch.cat((x_vis_p, self.vis_rgbd1(x_vis_p)), dim=1)
        x_vis_p2 = torch.cat((x_vis_p1, self.vis_rgbd2(x_vis_p1)), dim=1)
        x_vis_p3 = torch.cat((x_vis_p2, self.vis_rgbd3(x_vis_p2)), dim=1)

        x_inf_p=self.inf_conv(x_inf_origin)
        x_inf_p1 = torch.cat((x_inf_p, self.inf_rgbd1(x_inf_p)), dim=1)
        x_inf_p2 = torch.cat((x_inf_p1, self.inf_rgbd2(x_inf_p1)), dim=1)
        x_inf_p3 = torch.cat((x_inf_p2, self.inf_rgbd3(x_inf_p2)), dim=1)

        add_weight1 = torch.sigmoid(self.vis_weight1(x_vis_p1+x_inf_p1))
        fusion_img1  = add_weight1 * x_vis_p1 + (1 - add_weight1) * x_inf_p1

        add_weight2 = torch.sigmoid(self.vis_weight2(x_vis_p2+x_inf_p2))
        fusion_img2  = add_weight2 * x_vis_p2 + (1 - add_weight2) * x_inf_p2

        add_weight3 = torch.sigmoid(self.vis_weight3(x_vis_p3+x_inf_p3))
        fusion_img3  = add_weight3 * x_vis_p3 + (1 - add_weight3) * x_inf_p3

        # x_inf_p3=self.inf_rgbd3(x_inf_p2)
        # decode
        x=self.decode4(fusion_img3)
        x=self.decode3(fusion_img2+x)
        x=self.decode2(x+fusion_img1+x)
        x=self.decode1(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # 点乘（即“ * ”） ---- 各个矩阵对应元素做乘法;  区别于矩阵乘  .dot（）

class OptimizedDRFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, G=4):
        super(OptimizedDRFBlock, self).__init__()
        self.features = out_channels // 4

        # 初始卷积改用深度可分离卷积
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, self.features, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.features),
            nn.ReLU(inplace=True)
        )

        # 多分支卷积改用分组深度可分离卷积
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d((i + 1) * self.features, (i + 1) * self.features,
                          kernel_size=3, stride=1,
                          padding=d, dilation=d,
                          groups=G * (i + 1), bias=False),
                nn.Conv2d((i + 1) * self.features, self.features,
                          kernel_size=1, groups=G, bias=False),
                nn.BatchNorm2d(self.features),
                nn.ReLU(inplace=True)
            ) for i, d in enumerate([1, 2, 3])
        ])

        # SE注意力层保持不变
        self.se = SELayer(4 * self.features, 8)

        # 输出层使用深度可分离卷积
        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * self.features, 4 * self.features, kernel_size=1, groups=G, bias=False),
            nn.Conv2d(4 * self.features, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.conv0(x)

        # 多尺度特征提取
        x1 = torch.cat([x, self.dilated_convs[0](x)], dim=1)
        x2 = torch.cat([x1, self.dilated_convs[1](x1)], dim=1)
        x3 = torch.cat([x2, self.dilated_convs[2](x2)], dim=1)

        # 注意力机制
        feats = self.se(x3)

        # 输出
        return self.conv_out(feats)

class drf_block3(nn.Module):
    def __init__(self,in_channels,out_channels,G=4):
        super(drf_block3, self).__init__()
        self.features = out_channels//4

        self.convs_layer0 = Conv1(in_channels, self.features)

        self.convs_layer1 = nn.Sequential(
            nn.Conv2d(self.features , self.features, kernel_size=3, stride=1, padding=1, dilation=1, groups=G,
                      bias=False),
            nn.BatchNorm2d(self.features),
            nn.ReLU(inplace=False)
        )
        self.convs_layer2 = nn.Sequential(
            nn.Conv2d(2*self.features, self.features, kernel_size=3, stride=1, padding=2, dilation=2, groups=G,
                      bias=False),
            nn.BatchNorm2d(self.features),
            nn.ReLU(inplace=False),
        )
        self.convs_layer3 = nn.Sequential(
            nn.Conv2d(3*self.features, self.features, kernel_size=3, stride=1, padding=3, dilation=3, groups=G,
                      bias=False),
            nn.BatchNorm2d(self.features),
            nn.ReLU(inplace=False)
        )

        self.se = SELayer(4*self.features, 8)

        self.convs_layer4 = Conv1(4*self.features,out_channels)

    def forward(self,x):

        x = self.convs_layer0(x)

        x1 = torch.cat((x,self.convs_layer1(x)),dim=1)
        x2 = torch.cat((x1,self.convs_layer2(x1)), dim=1)
        x3 = torch.cat((x2, self.convs_layer3(x2)), dim=1)
        #feats = torch.cat((x1,x2,x3), dim=1)
        feats_U = self.se(x3)

        feats_U = self.convs_layer4(feats_U)
        # feats = feats.view(batch_size, 3, self.features, feats.shape[2], feats.shape[3])
        # feats_U = torch.sum(feats, dim=1)
        return feats_U


class ChannelAttentionFusion(nn.Module):
    def __init__(self, channels, num_groups=6):
        super().__init__()
        self.num_groups = num_groups
        # 通道注意力网络：为每组特征生成权重
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),  # 输入维度需为 (B*6, C)
            nn.ReLU(),
            nn.Linear(channels // 4, 1),  # 输出单个权重值
            nn.Softmax(dim=1)  # 对组维度归一化
        )

    def forward(self, features):
        # features: List[Tensor], 每个Tensor维度为 (B, C, H, W)
        # Step 1: 全局平均池化（压缩空间维度）
        gap_features = [torch.mean(f, dim=(2, 3)) for f in features]  # 每个元素维度 (B, C)

        # Step 2: 拼接并生成注意力权重
        stacked_gap = torch.stack(gap_features, dim=1)  # 维度 (B, 6, C)
        B, G, C = stacked_gap.shape
        # 将 (B, 6, C) 展平为 (B*6, C)
        flattened_gap = stacked_gap.view(B * G, C)
        # 通过全连接层生成权重
        attn_scores = self.attention(flattened_gap)  # 输出维度 (B*6, 1)
        attn_weights = attn_scores.view(B, G, 1, 1, 1)  # 恢复为 (B, 6, 1, 1, 1)

        # Step 3: 加权融合
        stacked_features = torch.stack(features, dim=1)  # 维度 (B, 6, C, H, W)
        fused_feature = torch.sum(attn_weights * stacked_features, dim=1)  # 输出维度 (B, C, H, W)
        return fused_feature

class Bimodal_Fusion_tmj(nn.Module):
    def __init__(self,in_channel=16):
        super().__init__()
        self.vis_weight1 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight2 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight3 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight4 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight5 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight6 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight7 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight8 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight9 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.vis_weight10 = ConvLeakyRelu2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.WeightedSumFusion = ChannelAttentionFusion(channels=in_channel, num_groups=6)

    def fusion_Or(self, x_vis, x_inf):
        fusion_Or_channel = torch.tanh(self.vis_weight5(x_vis) * self.vis_weight6(x_inf))\
                            + torch.tanh(self.vis_weight7(x_vis))+ torch.tanh(self.vis_weight8(x_inf))

        return fusion_Or_channel


    def fusion_and(self, x_vis, x_inf):
        fusion_and_channel = torch.tanh(self.vis_weight9(x_vis) * self.vis_weight10(x_inf))

        return fusion_and_channel

    def fusion_I_enhancement_V(self, x_vis, x_inf):
        fusion_I_V_enhancement = torch.tanh(self.vis_weight1(x_vis)) * torch.sigmoid(self.vis_weight2(x_inf))

        return fusion_I_V_enhancement

    def fusion_V_enhancement_I(self, x_vis, x_inf):
        fusion_V_I_enhancement = torch.tanh(self.vis_weight3(x_inf)) * torch.sigmoid(self.vis_weight4(x_vis))
        return fusion_V_I_enhancement


    def forward(self, x_vis, x_inf):

        fusion_Or_channel = self.fusion_Or(x_vis, x_inf)
        fusion_and_channel = self.fusion_and(x_vis, x_inf)
        fusion_I_enhancement_V = self.fusion_I_enhancement_V(x_vis, x_inf)
        fusion_V_enhancement_I = self.fusion_V_enhancement_I(x_vis, x_inf)
        fusion_result = self.WeightedSumFusion([fusion_Or_channel,fusion_and_channel,fusion_I_enhancement_V,fusion_V_enhancement_I,x_vis, x_inf])
        return fusion_result

class LSTM_Update(nn.Module):
    def __init__(self, center_size=9, off_center_size=16):
        super().__init__()
        self.off_center_size = off_center_size
        self.center_size = center_size
        self.W = nn.Parameter(torch.Tensor(off_center_size, center_size * 3))
        self.U = nn.Parameter(torch.Tensor(center_size, center_size * 3))
        self.bias = nn.Parameter(torch.Tensor(center_size * 3))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.center_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, center , off_center ):

        c_t, x_t  = center , off_center

        HS = self.center_size

        # batch the computations into a single matrix multiplication
        gates = x_t @ self.W + c_t @ self.U + self.bias
        i_t, f_t, g_t = (
            torch.sigmoid(gates[:, :HS]),  # input
            torch.sigmoid(gates[:, HS:HS * 2]),  # forget
            torch.tanh(gates[:, HS * 2:HS * 3]),
        )
        c_t = f_t * c_t + i_t * g_t

        return c_t, x_t

class Custom_On_Off_Center_filters_Conv(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, kernel_size=5, ratio = 1.5 ,off=False):
        """
        新增 kernel_size 参数，支持 5, 7, 9 等奇数尺寸
        """
        super().__init__()
        self.kernel_size = kernel_size

        # 自动计算 radius 和 gamma
        gamma = 1.0 / ratio
        radius = ((kernel_size + 1) / 2.0) * gamma

        # 自动推导 center_side (近似为 2*radius，并确保是奇数)
        # 例如 radius=2.0 -> side=3; radius=2.66 -> side=5
        estimated_side = int(radius * 2)
        if estimated_side % 2 == 0:
            estimated_side -= 1  # 强制转为奇数
        self.center_side = max(1, estimated_side)  # 至少为 1

        conv_On_filters = On_Off_Center_filters(radius=radius, gamma=gamma,
                                                in_channels=in_channels, out_channels=out_channels, off=off)
        inti_conv_On_filters = torch.tensor(conv_On_filters).permute(3, 2, 0, 1)
        self.kernel = nn.Parameter(inti_conv_On_filters.clone()).cuda()

        # 动态创建中心和外周索引掩码 (剥离最外层2圈作为外周)
        self.center_mask = torch.zeros(kernel_size, kernel_size, dtype=torch.bool)
        # 计算中心区域的起始和结束索引，使其居中
        # 例如 9x9核，5x5中心: start = (9-5)//2 = 2, end = 2+5 = 7. 切片为 [2:7, 2:7]
        start_idx = (kernel_size - self.center_side) // 2
        end_idx = start_idx + self.center_side
        self.center_mask[start_idx:end_idx, start_idx:end_idx] = True
        self.outer_mask = ~self.center_mask

        # 计算中心和外周的参数量
        self.center_dim = self.center_side * self.center_side
        self.outer_dim = kernel_size * kernel_size - self.center_dim

        # 动态初始化 LSTM_Update
        # 5x5: center_size=9,  off_center_size=16
        # 7x7: center_size=25, off_center_size=24
        # 9x9: center_size=25, off_center_size=56
        self.update_weight = LSTM_Update(center_size=self.center_dim, off_center_size=self.outer_dim)

    def forward(self, x):
        out_channels, in_channels, _, _ = self.kernel.shape

        """分解阶段"""
        # 提取中心参数
        center_params = self.kernel[:, :, self.center_mask].view(out_channels, in_channels, self.center_dim).permute(0,
                                                                                                                     2,
                                                                                                                     1).contiguous().view(
            -1, self.center_dim)

        # 提取外周参数
        outer_params = self.kernel[:, :, self.outer_mask].view(out_channels, in_channels, self.outer_dim).permute(0, 2,
                                                                                                                  1).contiguous().view(
            -1, self.outer_dim)

        """参数操作阶段"""
        modified_center, modified_outer = self.update_weight(center_params, outer_params)

        """重建阶段"""
        # 重建中心部分
        center_rebuilt = modified_center.view(out_channels, self.center_dim, in_channels).permute(0, 2, 1).view(
            out_channels, in_channels, self.center_side, self.center_side)

        # 重建外周部分
        outer_rebuilt = modified_outer.view(out_channels, self.outer_dim, in_channels).permute(0, 2, 1).view(
            out_channels, in_channels, self.outer_dim)

        # 创建新卷积核容器
        new_kernel = torch.zeros_like(self.kernel)
        # 填充中心区域
        new_kernel[:, :, self.center_mask] = center_rebuilt.reshape(out_channels, in_channels, self.center_dim)
        # 填充外周区域
        new_kernel[:, :, self.outer_mask] = outer_rebuilt.reshape(out_channels, in_channels, self.outer_dim)

        """执行卷积"""
        # 动态计算 padding 保持特征图尺寸不变 (例如 7x7 padding=3, 9x9 padding=4)
        pad_size = self.kernel_size // 2
        return F.conv2d(x, new_kernel, padding=pad_size)

class Bimodal_net(nn.Module):
    def __init__(self, init_weights=True):
        super(Bimodal_net, self).__init__()

        vis_ch = [16, 16, 16, 16]
        inf_ch = [16, 16, 16, 16]

        output = 1
        self.vis_conv = ConvLeakyRelu2d(1, vis_ch[0])
        self.vis_rgbd1 = drf_block3(vis_ch[0], vis_ch[1])
        self.vis_rgbd2 = drf_block3(vis_ch[1], vis_ch[2])
        self.vis_rgbd3 = drf_block3(vis_ch[2], vis_ch[3])

        self.inf_conv = ConvLeakyRelu2d(1, inf_ch[0])
        self.inf_rgbd1 = drf_block3(inf_ch[0], inf_ch[1])
        self.inf_rgbd2 = drf_block3(inf_ch[1], inf_ch[2])
        self.inf_rgbd3 = drf_block3(inf_ch[2], inf_ch[3])

        # self.decode5 = ConvBnLeakyRelu2d(vis_ch[3]+inf_ch[3], vis_ch[2]+inf_ch[2])
        self.decode4 = nn.Sequential(
                        ConvBnRelu2d(vis_ch[3], vis_ch[2]),
                        ConvBnRelu2d(vis_ch[2], vis_ch[2]),
                        ConvBnRelu2d(vis_ch[2], vis_ch[2])
        )
        self.decode3 = nn.Sequential(
                        ConvBnRelu2d(vis_ch[2], vis_ch[1]),
                        ConvBnRelu2d(vis_ch[1], vis_ch[1]),
                        ConvBnRelu2d(vis_ch[1], vis_ch[1]))
        self.decode2 = nn.Sequential(
                        ConvBnRelu2d(vis_ch[1], vis_ch[0]),
                        ConvBnRelu2d(vis_ch[0], vis_ch[0]),
                        ConvBnRelu2d(vis_ch[0], vis_ch[0]))
        self.decode1 = nn.Sequential(
                        ConvBnRelu2d(vis_ch[0], vis_ch[0]),
                        ConvBnRelu2d(vis_ch[0], vis_ch[0]),
                        ConvBnRelu2d(vis_ch[0], output))

        self.conv_On_Off_Center_filters_vi3 = nn.Sequential(
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=True),
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=False),
        )
        self.conv_On_Off_Center_filters_ir3 = nn.Sequential(
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=True),
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=False),
        )
        self.conv_On_Off_Center_filters_vi2 = nn.Sequential(
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=True),
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=False),
        )
        self.conv_On_Off_Center_filters_ir2 = nn.Sequential(
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=True),
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=False),
        )
        self.conv_On_Off_Center_filters_vi1 = nn.Sequential(
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=True),
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=False),
        )
        self.conv_On_Off_Center_filters_ir1 = nn.Sequential(
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=True),
            Custom_On_Off_Center_filters_Conv(in_channels=16, out_channels=16, off=False),
        )

        self.Bimodal_Fusion3 = Bimodal_Fusion_tmj(in_channel=vis_ch[3])
        self.Bimodal_Fusion2 = Bimodal_Fusion_tmj(in_channel=vis_ch[2])
        self.Bimodal_Fusion1 = Bimodal_Fusion_tmj(in_channel=vis_ch[1])

        if init_weights:
            self.__init_weights()

    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))



    def forward(self, image_vis, image_ir):

        x_vis_origin = image_vis
        x_inf_origin = image_ir

        # st = time.time()
        x_vis_p = self.vis_conv(x_vis_origin)
        x_vis_p1 = self.vis_rgbd1(x_vis_p)
        x_vis_p2 = self.vis_rgbd2(x_vis_p1)
        x_vis_p3 = self.vis_rgbd3(x_vis_p2)

        x_inf_p = self.inf_conv(x_inf_origin)
        x_inf_p1 = self.inf_rgbd1(x_inf_p)
        x_inf_p2 = self.inf_rgbd2(x_inf_p1)
        x_inf_p3 = self.inf_rgbd3(x_inf_p2)

        x_vis_p3_res = self.conv_On_Off_Center_filters_vi3[0](x_vis_p3) + self.conv_On_Off_Center_filters_vi3[1](x_vis_p3)
        x_inf_p3_res  = self.conv_On_Off_Center_filters_ir3[0](x_inf_p3) + self.conv_On_Off_Center_filters_ir3[1](x_inf_p3)
        fusion_img3 = self.Bimodal_Fusion3(x_vis_p3+x_vis_p3_res,x_inf_p3+x_inf_p3_res)

        x_vis_p2_res = self.conv_On_Off_Center_filters_vi2[0](x_vis_p2)+self.conv_On_Off_Center_filters_vi2[1](x_vis_p2)
        x_inf_p2_res = self.conv_On_Off_Center_filters_ir2[0](x_inf_p2)+self.conv_On_Off_Center_filters_ir2[1](x_inf_p2)
        fusion_img2 = self.Bimodal_Fusion2(x_vis_p2+x_vis_p2_res, x_inf_p2+x_inf_p2_res)

        x_vis_p1_res = self.conv_On_Off_Center_filters_vi1[0](x_vis_p1)+self.conv_On_Off_Center_filters_vi1[1](x_vis_p1)
        x_inf_p1_res = self.conv_On_Off_Center_filters_ir1[0](x_inf_p1) + self.conv_On_Off_Center_filters_ir1[1](x_inf_p1)
        fusion_img1 = self.Bimodal_Fusion1(x_vis_p1+x_vis_p1_res, x_inf_p1+x_inf_p1_res)

        x = self.decode4(fusion_img3)
        x = self.decode3(x+fusion_img2)
        x = self.decode2(x+fusion_img1)
        x = self.decode1(x)

        return x

    def sorround_modulation_DoG_on(self, input):
        filter_weights = torch.tensor(self.conv_On_filters).permute(3,2,0,1).cuda()
        # conv_On_filters=On_Off_Center_filters(radius=2.0, gamma=2. / 3., in_channels=16, out_channels=16,off=False)
        #
        # filter_weights =  torch.tensor(conv_On_filters).permute(3, 2, 0, 1).cuda()
        output = F.conv2d(input=input, weight=filter_weights, stride=(1, 1), padding='same')
        return output

    def sorround_modulation_DoG_off(self, input):
        filter_weights = torch.tensor(self.conv_Off_filters).permute(3,2,0,1).cuda()
        output = F.conv2d(input=input, weight=filter_weights, stride=(1, 1), padding='same')
        return output

class DualUNetFusion(nn.Module):
    def __init__(self, fusion_method='add', out_ch=1):
        super(DualUNetFusion, self).__init__()
        base = 16  # 通道基数
        self.fusion_method = fusion_method
        self.base = base

        # VIS Encoder
        self.vis_conv1 = self._make_encoder_block(1, base)
        self.vis_conv2 = self._make_encoder_block(base, base)
        self.vis_conv3 = self._make_encoder_block(base, base)

        # IR Encoder
        self.ir_conv1 = self._make_encoder_block(1, base)
        self.ir_conv2 = self._make_encoder_block(base, base)
        self.ir_conv3 = self._make_encoder_block(base, base)

        # Decoder
        self.decode4 = self._make_decoder_block(base, base)
        self.decode3 = self._make_decoder_block(base, base)
        self.decode2 = self._make_decoder_block(base, base)

        self.decode1 = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, out_ch, 1)
        )

        # Fusion modules (optional conv for concat)
        if fusion_method == 'concat':
            self.fuse1 = nn.Sequential(nn.Conv2d(base * 2, base, 1), nn.ReLU(inplace=True))
            self.fuse2 = nn.Sequential(nn.Conv2d(base * 2, base, 1), nn.ReLU(inplace=True))
            self.fuse3 = nn.Sequential(nn.Conv2d(base * 2, base, 1), nn.ReLU(inplace=True))

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self._init_weights()

    # Encoder block
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    # Decoder block
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    # Fusion function
    def _fuse(self, x1, x2, level):
        if self.fusion_method == 'add':
            return x1 + x2
        elif self.fusion_method == 'concat':
            x = torch.cat([x1, x2], dim=1)
            if level == 1:
                return self.fuse1(x)
            elif level == 2:
                return self.fuse2(x)
            elif level == 3:
                return self.fuse3(x)
        else:
            raise NotImplementedError(f"Fusion method '{self.fusion_method}' not supported.")

    # Weight initialization
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vis, ir):
        # Encoder
        v1 = self.vis_conv1(vis)
        i1 = self.ir_conv1(ir)
        f1 = self._fuse(v1, i1, level=1)

        v2 = self.vis_conv2(self.pool(v1))
        i2 = self.ir_conv2(self.pool(i1))
        f2 = self._fuse(v2, i2, level=2)

        v3 = self.vis_conv3(self.pool(v2))
        i3 = self.ir_conv3(self.pool(i2))
        f3 = self._fuse(v3, i3, level=3)

        # Decoder
        d4 = self.decode4(f3)

        # --- [核心修改 1] ---
        up_d4 = self.up(d4)
        # 检查上采样后的尺寸是否与跳跃连接的特征图 f2 匹配
        if up_d4.shape != f2.shape:
            # 如果不匹配，进行填充。F.pad参数: (左, 右, 上, 下)
            pad_H = f2.shape[2] - up_d4.shape[2]
            pad_W = f2.shape[3] - up_d4.shape[3]
            up_d4 = F.pad(up_d4, (0, pad_W, 0, pad_H))
        d3 = self.decode3(up_d4 + f2)

        # --- [核心修改 2] ---
        up_d3 = self.up(d3)
        # 检查上采样后的尺寸是否与跳跃连接的特征图 f1 匹配
        if up_d3.shape != f1.shape:
            # 如果不匹配，进行填充
            pad_H = f1.shape[2] - up_d3.shape[2]
            pad_W = f1.shape[3] - up_d3.shape[3]
            up_d3 = F.pad(up_d3, (0, pad_W, 0, pad_H))
        d2 = self.decode2(up_d3 + f1)

        d1 = self.decode1(d2)

        return d1

    def forward1(self, vis, ir):
        # Encoder
        v1 = self.vis_conv1(vis)
        i1 = self.ir_conv1(ir)
        f1 = self._fuse(v1, i1, level=1)

        v2 = self.vis_conv2(self.pool(v1))
        i2 = self.ir_conv2(self.pool(i1))
        f2 = self._fuse(v2, i2, level=2)

        v3 = self.vis_conv3(self.pool(v2))
        i3 = self.ir_conv3(self.pool(i2))
        f3 = self._fuse(v3, i3, level=3)

        # Decoder
        d4 = self.decode4(f3)
        d3 = self.decode3(self.up(d4) + f2)
        d2 = self.decode2(self.up(d3) + f1)
        d1 = self.decode1(d2)

        return d1

class DenseFuseModified(nn.Module):
    def __init__(self, base_channels=16, fusion_type='add'):
        super(DenseFuseModified, self).__init__()
        self.fusion_type = fusion_type
        self.base = base_channels

        # Encoder for both inputs (shared structure)
        self.encoder_initial = nn.Sequential(
            nn.Conv2d(1, self.base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.base, self.base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(self.base, self.base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Fusion doesn't use external functions, handled in forward
        # Decoder
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(self.base, self.base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(self.base, self.base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_output = nn.Conv2d(self.base, 1, kernel_size=1)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        x = self.encoder_initial(x)
        x = self.encoder_conv1(x)
        x = self.encoder_conv2(x)
        return x

    def fuse(self, feat1, feat2):
        if self.fusion_type == 'add':
            return (feat1 + feat2) / 2
        elif self.fusion_type == 'l1':
            abs1 = torch.abs(feat1)
            abs2 = torch.abs(feat2)
            mask = (abs1 >= abs2).float()
            return feat1 * mask + feat2 * (1 - mask)
        else:
            raise NotImplementedError(f"Fusion type '{self.fusion_type}' not supported")

    def decode(self, fused):
        x = self.decoder_conv1(fused)
        x = self.decoder_conv2(x)
        x = self.decoder_output(x)
        return x

    def forward(self, vis, ir):
        # Encode visual and infrared
        vis_feat = self.encode(vis)
        ir_feat = self.encode(ir)

        # Fuse features
        fused_feat = self.fuse(vis_feat, ir_feat)

        # Decode to fused image
        out = self.decode(fused_feat)

        # Ensure output size matches input
        if out.shape[2:] != vis.shape[2:]:
            out = F.interpolate(out, size=vis.shape[2:], mode='bilinear', align_corners=False)

        return out

class DenseFuse(nn.Module):
    """
    一个完全自包含的、忠实复现原始 DenseFuse 论文架构的 PyTorch 实现。

    此版本将辅助的 DenseBlock 类作为内部嵌套类 `_DenseBlock` 实现，
    从而将所有代码逻辑都封装在 `DenseFuse` 这一个类中。

    核心特性:
    1. 编码器 (Encoder): 共享权重，包含一个4层结构的内部密集连接块 `_DenseBlock`。
    2. 融合层 (Fusion Layer): 支持 'add' (加法) 和 'l1' (L1范数) 策略。
    3. 解码器 (Decoder): 一个标准的4层卷积网络，用于从融合特征重建图像。
    """

    # --- 1. 内部辅助类定义 ---
    class _DenseBlock(nn.Module):
        """
        内部辅助模块：实现单个密集连接块 (Conv + ReLU)。
        作为嵌套类，其实现细节被封装在主模型内部。
        """

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.conv(x))

    # --- 2. 主模型初始化 ---
    def __init__(self, base_channels=16, fusion_type='add'):
        super(DenseFuse, self).__init__()

        if fusion_type not in ['add', 'l1']:
            raise ValueError(f"不支持的融合类型: {fusion_type}. 请选择 'add' 或 'l1'.")

        self.fusion_type = fusion_type
        self.base = base_channels

        # --- 2.1. 共享权重的编码器 (Encoder) ---
        # 使用内部定义的 _DenseBlock 构建编码器。
        # 根据原论文结构 (Fig. 2)，这是一个4层的密集连接块。
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1)
        self.encoder_conv1 = self._DenseBlock(self.base, self.base)
        self.encoder_conv2 = self._DenseBlock(self.base * 2 , self.base)
        self.encoder_conv3 = self._DenseBlock(3 * self.base, self.base)

        # 编码器总输出通道数 = 4 * base_channels
        encoder_output_channels = 4 * self.base

        # --- 2.2. 解码器 (Decoder) ---
        # 解码器从融合特征中重建图像，是一个标准的4层CNN。
        self.decoder_conv1 = nn.Conv2d(encoder_output_channels, self.base * 3, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(self.base * 3, self.base * 2, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(self.base * 2, self.base, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv2d(self.base, 1, kernel_size=3, padding=1)# 最后一层输出单通道图像

        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        """
        实现编码器的密集连接逻辑。
        每一层的输入都是原始输入和所有先前特征图的拼接。
        """
        x_initial = x

        # 第1层
        feat = self.conv1(x_initial)
        feat1 = self.encoder_conv1(feat)
        feat2 = self.encoder_conv2(torch.cat([feat, feat1], dim=1))
        feat3 = self.encoder_conv3(torch.cat([feat, feat1, feat2], dim=1))

        # 最终编码器输出是所有特征图的拼接
        return torch.cat([feat, feat1, feat2, feat3], dim=1)

    def fuse(self, feat_vis, feat_ir):
        """融合来自两个编码器的特征图"""
        if self.fusion_type == 'add':
            # 论文中的主要方法是简单相加
            return (feat_vis + feat_ir)/2

        elif self.fusion_type == 'l1':
            # 基于L1范数的活动水平度量和选择策略
            # 1. 计算活动水平图 (沿通道维度的L1范数)
            activity_vis = torch.sum(torch.abs(feat_vis), dim=1, keepdim=True)
            activity_ir = torch.sum(torch.abs(feat_ir), dim=1, keepdim=True)

            # 2. 创建一个掩码，其中可见光特征活动水平更高的地方为1
            mask = (activity_vis >= activity_ir).float()

            # 3. 使用掩码组合特征
            fused_feat = feat_vis * mask + feat_ir * (1 - mask)
            return fused_feat

    def decode(self, fused_feat):
        """使用4层CNN从融合特征中重建图像"""
        x = self.relu(self.decoder_conv1(fused_feat))
        x = self.relu(self.decoder_conv2(x))
        x = self.relu(self.decoder_conv3(x))
        x = self.relu(self.decoder_conv4(x))
        return x

    def forward(self, vis, ir):
        """
        DenseFuse模型的主前向传播过程。
        vis: 可见光图像张量 (B, 1, H, W)
        ir: 红外图像张量 (B, 1, H, W)
        """
        # 1. 使用共享权重的密集编码器分别编码两个图像
        feat_vis = self.encode(vis)
        feat_ir = self.encode(ir)

        # 2. 融合提取出的特征
        fused_feat = self.fuse(feat_vis, feat_ir)

        # 3. 解码融合特征以重建最终图像
        output = self.decode(fused_feat)

        return output

