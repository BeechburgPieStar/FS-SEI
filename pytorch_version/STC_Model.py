from torch import nn
import torch


class ComplexConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=None):
        super(ComplexConv1D, self).__init__()
        self.padding = padding
        self.activation = nn.ReLU() if activation == 'relu' else None
        assert in_channels % 2 == 0, "The number of input channels must be even."
        in_channels = in_channels // 2
        out_channels = out_channels // 2
        # Model components
        self.conv_re = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # shape of x : [batch, channel, axis1]
        x_real = x[:, 0:x.shape[1] // 2, :]
        x_img = x[:, x.shape[1] // 2: x.shape[1], :]
        real = self.conv_re(x_real) - self.conv_im(x_img)
        imaginary = self.conv_re(x_img) + self.conv_im(x_real)
        outputs = torch.cat((real, imaginary), dim=1)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class Encoder(nn.Module):
    def __init__(self, feature_dim=1024):
        super(Encoder, self).__init__()
        self.conv1 = ComplexConv1D(2, 64, 3, padding=1, activation='relu')
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = ComplexConv1D(64, 64, 3, padding=1, activation='relu')
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = ComplexConv1D(64, 64, 3, padding=1, activation='relu')
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = ComplexConv1D(64, 64, 3, padding=1, activation='relu')
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = ComplexConv1D(64, 64, 3, padding=1, activation='relu')
        self.bn5 = nn.BatchNorm1d(64)
        self.pool5 = nn.MaxPool1d(2)
        self.conv6 = ComplexConv1D(64, 64, 3, padding=1, activation='relu')
        self.bn6 = nn.BatchNorm1d(64)
        self.pool6 = nn.MaxPool1d(2)
        self.conv7 = ComplexConv1D(64, 64, 3, padding=1, activation='relu')
        self.bn7 = nn.BatchNorm1d(64)
        self.pool7 = nn.MaxPool1d(2)
        self.conv8 = ComplexConv1D(64, 64, 3, padding=1, activation='relu')
        self.bn8 = nn.BatchNorm1d(64)
        self.pool8 = nn.MaxPool1d(2)
        self.conv9 = ComplexConv1D(64, 64, 3, padding=1, activation='relu')
        self.bn9 = nn.BatchNorm1d(64)
        self.pool9 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(576, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.bn1(self.conv1(x)))
        x = self.pool2(self.bn2(self.conv2(x)))
        x = self.pool3(self.bn3(self.conv3(x)))
        x = self.pool4(self.bn4(self.conv4(x)))
        x = self.pool5(self.bn5(self.conv5(x)))
        x = self.pool6(self.bn6(self.conv6(x)))
        x = self.pool7(self.bn7(self.conv7(x)))
        x = self.pool8(self.bn8(self.conv8(x)))
        x = self.pool9(self.bn9(self.conv9(x)))
        x = self.relu(self.fc(self.flatten(x)))
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes=30, feature_dim=1024):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class STC_Model(nn.Module):
    def __init__(self, n_classes=30, feature_dim=1024):
        super(STC_Model, self).__init__()
        self.encoder = Encoder(feature_dim)
        self.classifier = Classifier(n_classes, feature_dim)

    def forward(self, x):
        f = self.encoder(x)
        x = self.classifier(f)
        return f, x
