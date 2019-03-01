class DNet(torch.nn.Module):
    """
    Discriminator network:
    Inputs: Images from dataset as flattened vectors
    Architecture: 3 hidden layers with LReLU & Dropout
                  Sigmoid applied to output
    Returns: Probability of input belonging to real dataset
    """
    def __init__(self):
        super(DNet, self).__init__()
        n_features = 784 # 28x28 input image = 784 flat vector
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, n_out), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class GNet(torch.nn.Module):
    """
    Generator network:
    Inputs: 100 dim. vector of Gaussian noise: Variable(torch.randn(size, 100))
    Architecture: 3 hidden layers with LReLU & Dropout
                  Hyperbolic Tangent applied to output
    Outputs: Vector of length 784 containing generated data
    """
    def __init__(self):
        super(GNet, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256), nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512), nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024), nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, n_out), nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
