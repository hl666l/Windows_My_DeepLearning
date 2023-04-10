import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=6,
                            kernel_size=5,
                            padding=2,
                            stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=6,
                            out_channels=16,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=8,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  # 8x26x26
        )
        self.linear1 = torch.nn.Linear(5408, 1352)
        self.linear2 = torch.nn.Linear(1352, 338)
        self.linear3 = torch.nn.Linear(338, 120)
        self.linear4 = torch.nn.Linear(120, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
