from torch import nn

CLASSES = ['airplane', 'alarm clock', 'banana', 'baseball bat', 'bicycle', 'candle', 'car', 'crown',
           'dumbbell', 'eye', 'fish', 'flower', 'hat', 'headphones', 'ice cream', 'knife', 'pants',
           'shoe', 'umbrella', 'windmill']

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14x5
            nn.Conv2d(5, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()  # 7x7x16
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 16, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 23)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
