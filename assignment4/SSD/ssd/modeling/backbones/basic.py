from torch import conv2d, nn
from typing import Tuple, List


class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    filter_size = 3
    conv_stride = 1
    conv_padding = 1

    pool_size = 2
    pool_stride = 2


    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        # My code:
        
        self.first_layer = nn.Sequential(
            nn.Conv2d(image_channels, 32, self.filter_size, self.conv_stride, self.conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size, self.pool_stride),
            nn.Conv2d(32, 64, self.filter_size, self.conv_stride, self.conv_padding),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size, self.pool_stride),
            nn.Conv2d(64, 64, self.filter_size, self.conv_stride, self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(64, self.out_channels[0], self.filter_size, 2, self.conv_padding),
            nn.ReLU(),
        )
        
        self.second_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_channels[0], 128, self.filter_size, self.conv_stride, self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(128, self.out_channels[1], self.filter_size, 2, self.conv_padding),
            nn.ReLU(),
        )
 
        self.third_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_channels[1], 256, self.filter_size, self.conv_stride, self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(256, self.out_channels[2], self.filter_size, 2, self.conv_padding),
            nn.ReLU(),
        )
        
        self.fourth_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_channels[2], 128, self.filter_size, self.conv_stride, self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(128, self.out_channels[3], self.filter_size, 2, self.conv_padding),
            nn.ReLU(),
        )
        
        self.fifth_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_channels[3], 128, self.filter_size, self.conv_stride, self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(128, self.out_channels[4], self.filter_size, 2, self.conv_padding),
            nn.ReLU(),
        )
        
        self.sixth_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.out_channels[4], 128, self.filter_size, self.conv_stride, self.conv_padding),
            nn.ReLU(),
            nn.Conv2d(128, self.out_channels[5], self.filter_size, self.conv_stride, 0),
            nn.ReLU(),
        )
        
        self.layers = [self.first_layer, self.second_layer, self.third_layer, self.fourth_layer, self.fifth_layer, self.sixth_layer]
        

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        
        for layer in self.layers:
            x = layer(x)
            out_features.append(x)
        
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

