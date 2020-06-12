import os
import argparse

import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets
from models import *
from utils import mov_avg
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--architecture", type=str, default="dcgan",
                    help="The architecture to use, possible are \"dense\" and \"dcgan\"")


args = parser.parse_args()


if __name__ == "__main__":
    os.makedirs("images/{}".format(args.architecture), exist_ok=True)
    os.makedirs("data/mnist", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    # TO DO: pass channels and latent_dim as args
    channels = 1
    latent_dim = 100
    sample_interval = 400

    if args.architecture == "dcgan":
        img_size = 32
    elif args.architecture == "dense":
        img_size = 28
    else:
        raise ValueError("architecture should be one of \"dense\" and \"dcgan\"")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (channels, img_size, img_size)

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    adversarial_loss = torch.nn.BCELoss()

    print("Creating model...")
    if args.architecture == "dcgan":
        generator = GeneratorDCGAN(latent_dim, img_size, channels=channels).to(device)
        discriminator = DiscriminatorDCGAN(img_size, channels=1).to(device)

    elif args.architecture == "dense":
        generator = GeneratorDense(latent_dim, img_shape).to(device)
        discriminator = DiscriminatorDense(img_shape).to(device)
    else:
        raise ValueError("architecture should be one of \"dense\" and \"dcgan\"")

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    d_loss_list = []
    g_loss_list = []

    print("Begin training...")
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).to(device)
            fake = torch.Tensor(imgs.size(0), 1).fill_(0.0).to(device)

            real_imgs = imgs.to(device)

            #  Train Generator
            optimizer_G.zero_grad()

            z = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))).to(device)

            # compute G(z)
            gen_imgs = generator(z)

            # computer log(1 - D(G(z)))
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Compute -log(D(x))
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            # Computer -log(1-D(G(z)))
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/{}/{}.png".format(args.architecture, batches_done),
                           nrow=5, normalize=True)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

    torch.save(generator, "models/generator_{}.pth".format(args.architecture))
    torch.save(discriminator, "models/discriminator_{}.pth".format(args.architecture))

    d_loss = np.array(d_loss_list)
    g_loss = np.array(g_loss_list)

    np.save("log/d_loss_{}.npy".format(args.architecture), d_loss)
    np.save("log/g_loss_{}.npy".format(args.architecture), g_loss)

    d_loss_smooth = np.array(list(mov_avg(d_loss, 400)))
    g_loss_smooth = np.array(list(mov_avg(g_loss, 400)))

    plt.plot(d_loss_smooth, label='discriminator loss')
    plt.plot(g_loss_smooth, label='generator loss')
    plt.legend()
    plt.savefig("{}_loss.png".format(args.architecture))

