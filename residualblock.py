import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, activation, spect_norm=True, batch_norm=True):
        super().__init__()

        if input_size == output_size:
            shortcut = nn.Identity()
        else:
            shortcut = [spectral_norm(nn.Linear(input_size, output_size)) if spect_norm else
                        nn.Linear(input_size, output_size)]
            shortcut += batch_norm * [nn.BatchNorm1d(output_size)]

        block = [spectral_norm(nn.Linear(input_size, output_size)) if spect_norm else
                 nn.Linear(input_size, output_size)]
        block += batch_norm * [nn.BatchNorm1d(output_size)] + [activation]
        block += [spectral_norm(nn.Linear(input_size, output_size)) if spect_norm else
                  nn.Linear(input_size, output_size)]
        block += batch_norm * [nn.BatchNorm1d(output_size)]

        self.shortcut = nn.Sequential(*shortcut)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
