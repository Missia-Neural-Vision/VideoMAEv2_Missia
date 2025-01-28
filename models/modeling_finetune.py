# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    """Creates a configuration dictionary with default values for video model settings.

    This function creates a configuration dictionary with predefined default values
    for various model parameters like input dimensions, preprocessing settings, etc.
    The function allows overriding any default values through keyword arguments.

    Args:
        url (str, optional): URL for model weights. Defaults to ''.
        **kwargs: Additional keyword arguments to override default configuration values.

    Returns:
        dict: Configuration dictionary containing the following default key-value pairs:
            - url (str): URL for model weights
            - num_classes (int): Number of output classes (default: 400)
            - input_size (tuple): Expected input dimensions (C, H, W) (default: (3, 224, 224))
            - pool_size (None): Pooling size
            - crop_pct (float): Crop percentage for preprocessing (default: 0.9)
            - interpolation (str): Interpolation method (default: 'bicubic')
            - mean (tuple): Normalization mean values (default: (0.5, 0.5, 0.5))
            - std (tuple): Normalization standard deviation values (default: (0.5, 0.5, 0.5))
    """
    return {
        'url': url,
        'num_classes': 400,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.

    This implements the DropPath regularization technique, also known as Stochastic Depth,
    which randomly drops entire paths (layers) during training for regularization.

    Args:
        drop_prob (float, optional): Probability of dropping a path. Defaults to None.

    Methods:
        forward(x): Applies drop path to the input tensor.
        extra_repr(): Returns string representation of drop probability.

    Returns:
        tensor: Output after applying drop path.

    Example:
        >>> m = DropPath(drop_prob=0.2)
        >>> output = m(input_tensor)

    References:
        Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    """Multi-layer Perceptron (MLP) module.

    A simple MLP that contains two linear transformations with a GELU activation in between.

    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden features. Defaults to in_features.
        out_features (int, optional): Number of output features. Defaults to in_features.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        drop (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        torch.Tensor: Output tensor after MLP transformation

    Example:
        >>> mlp = Mlp(in_features=768, hidden_features=3072)
        >>> x = torch.randn(1, 197, 768)
        >>> output = mlp(x)  # shape: (1, 197, 768)
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CosAttention(nn.Module):
    """Cosine Attention module.

    This module implements a multi-head attention mechanism using cosine similarity
    for computing attention scores. It normalizes the query and key vectors before
    computing their dot product and applies learned temperature scaling.

    Args:
        dim (int): Input dimension.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): If True, adds bias to query, key, value projections. 
            Defaults to False.
        qk_scale (float, optional): Override default qk scale of 1/sqrt(head_dim).
            Defaults to None.
        attn_drop (float, optional): Dropout rate for attention weights. 
            Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for output. Defaults to 0.0.
        attn_head_dim (int, optional): Override default head dimension calculation.
            Defaults to None.

    Attributes:
        num_heads (int): Number of attention heads
        scale (nn.Parameter or float): Learned or fixed scaling factor for attention scores
        qkv (nn.Linear): Linear projection for query, key, value
        q_bias (nn.Parameter or None): Bias for query projection
        v_bias (nn.Parameter or None): Bias for value projection
        proj (nn.Linear): Output projection
        attn_drop (nn.Dropout): Dropout for attention weights
        proj_drop (nn.Dropout): Dropout for output

    Returns:
        torch.Tensor: Transformed input tensor with same shape as input
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # self.scale = qk_scale or head_dim**-0.5
        # DO NOT RENAME [self.scale] (for no weight decay)
        if qk_scale is None:
            self.scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))),
                requires_grad=True)
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (
            F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        # torch.log(torch.tensor(1. / 0.01)) = 4.6052
        logit_scale = torch.clamp(self.scale, max=4.6052).exp()

        attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """Multi-head Self Attention module.

    This module implements multi-head self attention mechanism that allows the model to jointly attend
    to information from different representation subspaces at different positions.

    Args:
        dim (int): Input dimension.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): If True, use bias terms for query, key, value projections. Defaults to False.
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        proj_drop (float, optional): Dropout rate for output. Defaults to 0.0.
        attn_head_dim (int, optional): Override default head dimension if set. Defaults to None.

    Attributes:
        num_heads (int): Number of parallel attention heads.
        scale (float): Scaling factor for query-key attention scores.
        qkv (nn.Linear): Linear projection for query, key, and value.
        q_bias (nn.Parameter or None): Learnable bias for query projection.
        v_bias (nn.Parameter or None): Learnable bias for value projection.
        attn_drop (nn.Dropout): Dropout layer for attention weights.
        proj (nn.Linear): Output projection layer.
        proj_drop (nn.Dropout): Dropout layer for output.

    Returns:
        torch.Tensor: Attention output of shape (B, N, dim) where B is batch size,
            N is sequence length, and dim is input dimension.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """A Transformer block implementation for vision tasks.

    This block implements a standard Transformer architecture with self-attention
    and MLP layers, including optional layer scaling and stochastic depth.

    Args:
        dim (int): Input dimension/number of features.
        num_heads (int): Number of attention heads.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to input dimension. Defaults to 4.0.
        qkv_bias (bool, optional): If True, adds bias to the QKV projection. Defaults to False.
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
        drop (float, optional): Dropout rate. Defaults to 0.0.
        attn_drop (float, optional): Attention dropout rate. Defaults to 0.0.
        drop_path (float, optional): Stochastic depth rate. Defaults to 0.0.
        init_values (float, optional): Initial value for layer scale. If >0, enables layer scale. Defaults to None.
        act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        attn_head_dim (int, optional): Dimension of each attention head. Defaults to None.
        cos_attn (bool, optional): If True, uses cosine attention instead of dot product. Defaults to False.

    Attributes:
        norm1 (nn.Module): First normalization layer
        attn (nn.Module): Self-attention module (either standard or cosine)
        drop_path (nn.Module): Stochastic depth layer
        norm2 (nn.Module): Second normalization layer
        mlp (nn.Module): MLP module
        gamma_1 (nn.Parameter or None): Layer scale for attention output
        gamma_2 (nn.Parameter or None): Layer scale for MLP output

    Returns:
        torch.Tensor: Transformed input tensor of same shape as input
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 cos_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if cos_attn:
            self.attn = CosAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """3D Patch Embedding layer for video inputs.

    This module converts a video tensor into a sequence of patch embeddings. It first splits
    the video into non-overlapping 3D patches (tubelets) and then projects them into an
    embedding space using a 3D convolutional layer.

    Args:
        img_size (int, optional): Size of input image. Defaults to 224.
        patch_size (int, optional): Size of each patch. Defaults to 16.
        in_chans (int, optional): Number of input channels. Defaults to 3.
        embed_dim (int, optional): Dimension of output embeddings. Defaults to 768.
        num_frames (int, optional): Number of frames in input video. Defaults to 16.
        tubelet_size (int, optional): Temporal size of each 3D patch. Defaults to 2.

    Attributes:
        img_size (tuple): Spatial dimensions of input images (H, W).
        tubelet_size (int): Temporal dimension of 3D patches.
        patch_size (tuple): Spatial dimensions of patches (H, W).
        num_patches (int): Total number of patches.
        proj (nn.Conv3d): 3D convolution layer for patch projection.

    Shape:
        - Input: (B, C, T, H, W)
            B: batch size
            C: channels
            T: number of frames
            H: height
            W: width
        - Output: (B, N, D)
            N: number of patches
            D: embedding dimension
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 num_frames=16,
                 tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # b, c, l -> b, l, c
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Returns a sinusoidal positional encoding table as a tensor.

    The function creates a positional encoding matrix using sine and cosine functions
    at different frequencies, which helps transformer models encode sequential position
    information.

    Args:
        n_position (int): Number of positions to encode
        d_hid (int): Dimension of the positional encoding vectors

    Returns:
        torch.Tensor: A tensor of shape (1, n_position, d_hid) containing the
            positional encodings. The tensor has requires_grad=False.

    Note:
        This implementation currently uses numpy for the initial computation before
        converting to a torch tensor. The sine function is applied to even indices
        and cosine to odd indices of the encoding vectors.

    Example:
        >>> encoding_table = get_sinusoid_encoding_table(100, 512)
        >>> print(encoding_table.shape)
        torch.Size([1, 100, 512])
    """

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class VisionTransformer(nn.Module):
    """Vision Transformer model for video and image processing.

    This implementation supports patch or hybrid CNN input stage and includes various
    features like learnable/sinusoidal positional embeddings, dropout, and attention mechanisms.

    Args:
        img_size (int): Input image size. Default: 224
        patch_size (int): Size of patches to embed. Default: 16
        in_chans (int): Number of input channels. Default: 3
        num_classes (int): Number of classes for classification. Default: 1000
        embed_dim (int): Embedding dimension. Default: 768
        depth (int): Number of transformer blocks. Default: 12
        num_heads (int): Number of attention heads. Default: 12
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0
        qkv_bias (bool): Enable bias for qkv projections. Default: False
        qk_scale (float): Override default qk scale. Default: None
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Drop path rate. Default: 0.0
        head_drop_rate (float): Classifier head dropout rate. Default: 0.0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        init_values (float): Initial layer scale values. Default: 0.0
        use_learnable_pos_emb (bool): Use learnable positional embeddings. Default: False
        init_scale (float): Initial scale for classifier weights. Default: 0.0
        all_frames (int): Number of frames in input video. Default: 16
        tubelet_size (int): Size of video tubelets. Default: 2
        use_mean_pooling (bool): Use mean pooling for feature aggregation. Default: True
        with_cp (bool): Use checkpoint for gradient computation. Default: False
        cos_attn (bool): Use cosine attention. Default: False

    Attributes:
        num_classes (int): Number of output classes
        num_features (int): Number of output features (same as embed_dim)
        embed_dim (int): Dimension of token embeddings
        tubelet_size (int): Size of video tubelets
        
    Methods:
        forward_features(x): Computes features from input
        forward(x): Complete forward pass
        get_num_layers(): Returns number of transformer blocks
        get_classifier(): Returns classification head
        reset_classifier(num_classes): Resets the classification head
        
    Example:
        >>> model = VisionTransformer(img_size=224, patch_size=16, num_classes=1000)
        >>> x = torch.randn(1, 3, 16, 224, 224)  # [B, C, T, H, W]
        >>> output = model(x)  # [B, num_classes]
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 head_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 with_cp=False,
                 cos_attn=False):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn) for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
            embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.size(0)

        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(
                x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return self.norm(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_dropout(x)
        x = self.head(x)
        return x


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """Creates a small Vision Transformer (ViT) model with patch size 16x16 for 224x224 images.

    This function instantiates a ViT model with a specific configuration suitable for
    smaller-scale vision tasks. The model uses a patch size of 16x16 pixels and is designed
    for input images of size 224x224 pixels.

    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. Defaults to False.
        **kwargs: Additional arguments to pass to the VisionTransformer constructor.

    Returns:
        VisionTransformer: A Vision Transformer model with the following specifications:
            - Patch size: 16x16
            - Embedding dimension: 384
            - Depth: 12 transformer layers
            - Number of attention heads: 6
            - MLP ratio: 4
            - QKV bias: True
            - Normalization: LayerNorm with eps=1e-6

    Example:
        >>> model = vit_small_patch16_224(pretrained=False)
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """Initialize a base Vision Transformer model with patch size 16x16 and input size 224x224.

    This function creates a Vision Transformer (ViT) model with base configuration, which includes
    12 transformer layers, 12 attention heads, and embedding dimension of 768.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the VisionTransformer constructor.

    Returns:
        VisionTransformer: A Vision Transformer model instance with base configuration.

    Example:
        >>> model = vit_base_patch16_224(pretrained=False)
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """Creates a Vision Transformer (ViT) model with large architecture and 16x16 patches.

    This function instantiates a Vision Transformer model with specific architectural parameters
    suitable for 224x224 input images. The model uses patch size of 16, resulting in 14x14 patches.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
            Defaults to False.
        **kwargs: Additional arguments passed to the VisionTransformer constructor.

    Returns:
        VisionTransformer: A Vision Transformer model with the following specifications:
            - Patch size: 16x16
            - Embedding dimension: 1024
            - Depth: 24 transformer layers
            - Number of attention heads: 16
            - MLP ratio: 4
            - QKV bias: True
            - Layer normalization with eps=1e-6

    Example:
        >>> model = vit_large_patch16_224(pretrained=True)
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs):
    """Initialize a huge Vision Transformer model with patch size 16x16 for 224x224 images.

    This function creates a Vision Transformer (ViT) model with huge architecture specifications,
    designed for processing 224x224 pixel images with 16x16 patch size.

    Args:
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to False.
        **kwargs: Additional arguments to pass to the VisionTransformer constructor.

    Returns:
        VisionTransformer: A ViT model with the following specifications:
            - Patch size: 16x16
            - Embedding dimension: 1280
            - Depth: 32 layers
            - Number of attention heads: 16
            - MLP ratio: 4
            - QKV bias: True
            - Layer normalization epsilon: 1e-6

    Note:
        This model uses Layer Normalization with an epsilon value of 1e-6.
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    """Create a Giant Vision Transformer (ViT) model with patch size 14x14 for 224x224 images.

    This function creates a Vision Transformer model with the following specifications:
    - Patch size: 14x14
    - Embedding dimension: 1408
    - Depth: 40 layers
    - Number of attention heads: 16
    - MLP ratio: 48/11
    - QKV bias: True
    - Layer normalization epsilon: 1e-6

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        **kwargs: Additional arguments passed to the VisionTransformer constructor.

    Returns:
        VisionTransformer: A Giant Vision Transformer model instance.

    Note:
        This model variant follows the architecture described in the original ViT paper
        but with significantly larger parameters suitable for high-capacity tasks.
    """
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    return model
