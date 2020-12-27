import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

# setting the hyperparameters
batchsize = 64
imagesize = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.device_count

# Creating transformations
transform = transforms.Compose([transforms.Resize((imagesize,imagesize)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# Loading the dataset
dataset = dset.ImageFolder('C:/Users/neele/OneDrive/Documents/Datasets/bookimagesfortraining', transform=transform)
#dataset = dset.CIFAR10(root="./data", download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

# Defining the weights initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

# Define the generators
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.Generator_Sequential = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # Input would be a random vector of size 100
    def forward(self, input):   
        output = self.Generator_Sequential(input)
        return output

network_generator = Generator().to(device)
network_generator.apply(weights_init)

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.Discriminator_Sequential = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output =  self.Discriminator_Sequential(input)
        return output.view(-1)

network_discriminator = Discriminator().to(device)
network_discriminator.apply(weights_init)

criterion = nn.BCELoss()
optimizer_generator = torch.optim.Adam(network_generator.parameters(), 0.0002, (0.5,0.999))
optimizer_discriminator = torch.optim.Adam(network_discriminator.parameters(), 0.0002, (0.5,0.999))

# Training the GAN
epochs = 500

print("Starting with the training...")

for epoch in range(epochs):

    for i, data in enumerate(dataloader, 0):

        try:
        
            # Step 1 : Updating weights of neural network of the Discriminator
            network_discriminator.zero_grad()

            # Training the Discriminator with the real image from the dataset
            real, _ = data
            inputs = Variable(real.to(device))
            target = Variable(torch.ones(inputs.size()[0], device=device))
            output = network_discriminator(inputs)
            error_discriminator_real = criterion(output, target)

            # Training the Discriminator with the fake image from the Generator
            noise = Variable(torch.randn(inputs.size()[0], 100, 1, 1, device=device))
            fake = network_generator(noise)
            target = Variable(torch.zeros(inputs.size()[0], device=device))
            output = network_discriminator(fake.detach())
            error_discriminator_fake = criterion(output, target)

            # Backpropogating the total error
            error_discriminator = error_discriminator_real + error_discriminator_fake
            error_discriminator.backward() # Calculate the weights
            optimizer_discriminator.step() # Update the weights

            # Step 2 : Updating weights of neural network of the Generator
            network_generator.zero_grad()
            target = Variable(torch.ones(inputs.size()[0], device=device))
            output = network_discriminator(fake)
            error_generator = criterion(output, target)

            # Backpropogating the error of Generator
            error_generator.backward()
            optimizer_generator.step()

            # Visualize the training
            print('Training the model ==> [%d/%d][%d/%d] ----> Loss of the Discriminator : %.4f Loss of the Generator : %.4f' % (epoch, epochs, i, len(dataloader), error_discriminator.item(), error_generator.item()))

            # saving the generated image at every 100 step
            if i % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' % "./gan_results", normalize=True)
                fake = network_generator(noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./gan_results", epoch), normalize=True)
                
        except Exception as e:
            pass
    
    #print('Training the model ==> [%d/%d] ----> Loss of the Discriminator : %.4f Loss of the Generator : %.4f' % (epoch, epochs, error_discriminator.item(), error_generator.item()))

torch.save(network_generator, './Generator.pth')
torch.save(network_discriminator, './Discriminator.pth')
