import argparse
import os
import time
import pandas as pd
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch
from models import dcgan3D, gradnorm,Res_dcgan3D,SA_dcgan3D,Res_CBAM_dcgan3D
from utils.dataset import HDF5Dataset
from utils.hdf5_io import save_hdf5
from utils.functions import generate_dir2save


parser = argparse.ArgumentParser()
parser.add_argument('--input_name', default='/home1/Usr/yinmengkai/Datesets/dizhidata/berea_ti_hdf5',
                    help='input image name for training')
parser.add_argument('--dataroot', default='/home1/Usr/yinmengkai/Datesets/dizhidata/berea_ti_hdf5',
                    help='path to dataset')
parser.add_argument("--n_epochs", type=int, default=8000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")

parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")

parser.add_argument("--nz", type=int, default=100, help="Size of z latent vector")
parser.add_argument("--nc", type=int, default=1, help="Number of channels in the training images")

parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for dataloader")

parser.add_argument("--sample_interval", type=int, default=5, help="interval between image sampling")
parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
opt = parser.parse_args()
print(opt)

# For fast training
cudnn.benchmark = True
# 图像和模型输出路径
opt.dir2save = generate_dir2save(opt)

try:
    os.makedirs(opt.dir2save)
except OSError:
    pass

# 设备
device = torch.device('cuda:0')
# 开始时间
start = time.time()
# 初始化数据集
dataset = HDF5Dataset(opt.dataroot,
                      input_transform=transforms.Compose([
                          transforms.ToTensor()
                      ]))
# 加载数据集
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=int(opt.num_workers)
                                         )


# 在netG和netD上调用自定义权重初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 损失函数
criterion = torch.nn.BCEWithLogitsLoss()

# 初始化生成器和判别器
generator = Res_CBAM_dcgan3D.Generator(opt).to(device)
discriminator = Res_CBAM_dcgan3D.Discriminator(opt).to(device)

# 初始化权重
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

fixed_noise = torch.randn(1, opt.nz, 7, 7, 7).to(device)
fixed_noise_TI = torch.randn(1, opt.nz, 1, 1, 1).to(device)

#Tensor = torch.cuda.FloatTensor

# ----------
#  训练
# ----------

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        # 新建表头
        data = pd.DataFrame(columns=['epoch', 'epochs', 'i', 'dataloader_len', 'd_loss', 'loss_fake', 'D_x', 'D_G_z1', 'D_G_z2'])

        x_real = imgs.to(device)
        noise = torch.randn(opt.batch_size, opt.nz, 1, 1, 1, device=device)

        ############################
        # (1) 训练判别器
        ############################

        discriminator.zero_grad()

        x_fake = generator(noise)

        pred_real = gradnorm.normalize_gradient(discriminator, x_real)
        pred_fake = gradnorm.normalize_gradient(discriminator, x_fake)

        loss_real = criterion(pred_real, torch.ones_like(pred_real))
        loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

        d_loss = loss_real + loss_fake
        d_loss.backward()
        #
        D_x = pred_real.data.mean()
        #
        D_G_z1 = pred_fake.data.mean()
        optimizer_D.step()
        ############################
        # (2) 训练生成器
        ############################
        generator.zero_grad()

        x_fake = generator(noise)

        pred_fake = gradnorm.normalize_gradient(discriminator, x_fake)
        loss_fake = criterion(pred_fake, torch.ones_like(pred_fake))

        loss_fake.backward()
        #
        D_G_z2 = pred_fake.data.mean()
        optimizer_G.step()

        # 训练中损失值曲线
        # training_curve(epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), loss_fake.item(), D_x, D_G_z1, D_G_z2,opt)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x): %f] [D(G(z)): %f/%f] "
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), loss_fake.item(), D_x, D_G_z1, D_G_z2)
        )

        dit = {
            'epoch': epoch,
            'epochs': opt.n_epochs,
            'i': i,
            'len(dataloader)': len(dataloader),
            'd_loss': d_loss,
            'loss_fake': loss_fake,
            'D_x': D_x,
            'D_G_z1': D_G_z1,
            'D_G_z2': D_G_z2,
        }

        df = pd.DataFrame(dit, index=[0])
        data = data.append(df)
        data.to_excel(opt.dir2save + '/training_curve.xlsx', index=False)

    if epoch % opt.sample_interval == 0:
        fake = generator(fixed_noise)
        fake_TI = generator(fixed_noise_TI)
        # save_hdf5(fake.data, opt.dir2save + '/fake_samples_{0}.hdf5'.format(epoch))
        save_hdf5(fake_TI.data, opt.dir2save + '/fake_TI_{0}.hdf5'.format(epoch))

    # do checkpointing
    if epoch % opt.sample_interval == 0:
        torch.save(generator.state_dict(), opt.dir2save + '/netG_epoch_%d.pth' % (epoch))
        # torch.save(discriminator.state_dict(), opt.dir2save + '/netD_epoch_%d.pth' % (epoch))

end = time.time()
elapsed_time = end - start
print("运行时间：",elapsed_time)