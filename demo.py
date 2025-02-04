import torch
from torchvision import datasets, transforms
from nn import Model, Dense, Conv2D, MaxPool, Flatten, ReLU, CrossEntropy
from nn.optimizers import SGDOptimizer

# Define the model
model = Model()
model.add(Conv2D(input_size=[1], num_kernels=10, kernel_size=(5,5), padding=(0,0), stride=(1,1), activation=None))
model.add(MaxPool(pool_size=(2,2), stride=(2,2), activation=ReLU()))
model.add(Conv2D(input_size=[10], num_kernels=20, kernel_size=(5,5), padding=(0,0), stride=(1,1), activation=None))
model.add(MaxPool(pool_size=(2,2), stride=(2,2), activation=ReLU()))
model.add(Flatten())
model.add(Dense(320, 50, activation=ReLU()))
model.add(Dense(50, 10))
model.init_weights(0.01)
model.set_loss(CrossEntropy())

# Print the model summary
print(model.summary)

# Load the MNIST dataset
seed = 42
torch.manual_seed(seed)

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
    batch_size=batch_size, shuffle=True)

# Train the model
optim = SGDOptimizer(model.params, lr=0.001)
for epoch in range(3):
    print(f"Epoch: {epoch: 4}", "~" * 20)
    for batch_idx, (data, target) in enumerate(train_loader):
        data_numpy, target_numpy = data.numpy(), target.numpy()
        sample, target = data_numpy, target_numpy
        batch_size = sample.shape[0]
        # zero grad
        optim.zero_grad()
        # forward
        output = model(sample)
        # loss
        loss = model.loss(output, target)
        print("Loss:", loss)
        # backward
        model.backward()
        # update
        optim.step()
        print("~" * 32)