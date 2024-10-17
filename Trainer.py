import os
import random
import sys
import wandb
import yaml
import numpy as np
import argparse
from tqdm import tqdm
from matplotlib import transforms
from torchvision import utils as vutils
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.tensorboard import SummaryWriter
from DDQN import Duel_Q_Net, DQN_agent, BasicBuffer
from utils import evaluate_policy, str2bool

class EngageEnv:
    def __init__(self):
        self.state_ = []
        self.current_time = 0
        self.event_time = []

        self.event_list = []
        self.approach = []

    def get_next_step(self, actionIdx, current_time, task_queue, df):
        pass


class TrainDQN:
    def __init__(self, args):
        self.args = args
        self.current_epoch = 0

        self.nc = int(args.nc)
        self.nz = int(args.nz)
        self.ngf = int(args.ngf)
        self.ndf = int(args.ndf)

        self.netG = Generator(args, self.nc, self.nz, self.ngf).to(args.device)
        self.netG.apply(weights_init)
        if args.netG != "":
            self.netG.load_state_dict(torch.load(args.netG))
        print(self.netG)

        self.netD = Discriminator(args, self.nc, self.ndf).to(args.device)
        self.netD.apply(weights_init)
        if args.netD != "":
            self.netD.load_state_dict(torch.load(args.netD))
        print(self.netD)

        self.writer = SummaryWriter(args.log)
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))        
        
        if args.dry_run:
            args.epochs = 1

        self.prepare_training()
        self.G_losses = []
        self.D_losses = []

    def load_state_dict(self, state_dictg, state_dict2d):
        self.netG.load_state_dict(state_dictg['state_dict'])
        self.netD.load_state_dict(state_dict2d['state_dict'])
        self.last_epoch = state_dictg['last_epoch']

    @staticmethod
    def prepare_training():
        pass

    def train(self, train_loader, val_loader):
        for epoch in range(self.args.start_from_epoch + 1, self.args.epochs + 1):
            self.current_epoch = epoch
            self.train_one_epoch(epoch, train_loader)
            avg_acc = self.eval_one_epoch(epoch, val_loader)
            self.save(avg_acc)

    def train_one_epoch(self, epoch, train_loader):
        iters = 0
        total_loss_d = 0
        total_loss_g = 0
        self.netG.train()
        self.netD.train()
        for i, (image, cond) in enumerate(pbar := tqdm(train_loader)):
            # batch_size = 64, channels = 3, width = 64, height = 64

            self.optimizerD.zero_grad()
            self.optimizerG.zero_grad()

            device = args.device
            real_image = image.to(device)
            cond = cond.to(device)
            batch_size = image.size(0)

            self.writer.add_scalar('Train Step/Loss D', errD.item(), iters)
            self.writer.add_scalar('Train Step/Loss G', errG.item(), iters)
            self.writer.add_scalar('Train Step/D(x)', D_x, iters)
            self.writer.add_scalar('Train Step/D(G(z))', D_G_z1, iters)
            pbar.set_description('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, args.epochs, i, len(train_dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            pbar.refresh()
            total_loss_d += errD.item()
            total_loss_g += errG.item()
            if iters % 100 == 0:
                noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                vutils.save_image(real_image,
                    '%s/real_samples.png' % args.outf,
                    normalize=True)
                vutils.save_image(fake_image.detach(),
                    '%s/fake_samples.png' % args.outf,
                    normalize=True)
            iters += 1
        self.writer.add_scalar('Train Epoch/Loss D', total_loss_d/len(train_dataloader), epoch)
        self.writer.add_scalar('Train Epoch/Loss G', total_loss_g/len(train_dataloader), epoch)

    
    def tqdm_bar_train(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    @torch.no_grad()
    def eval_one_epoch(self, epoch, test_dataloader):
        args = self.args
        evaluator = evaluation_model()
        self.netG.eval()
        self.netD.eval()
        avg_acc = 0
        with torch.no_grad():
            for sample in range(10):
                for i, cond in enumerate(test_dataloader):
                    cond = cond.to(args.device)
                    batch_size = cond.size(0)
                    noise = torch.randn(batch_size, args.nz, 1, 1, device=args.device)
                    fake_image = self.netG(noise, cond)
                    vutils.save_image(fake_image.detach(),
                        f'{args.outf}/epoch_{epoch}_fake_test_sample_{sample}.png',
                        normalize=True)
                    acc = evaluator.eval(fake_image, cond)
                print(f'Sample {sample+1}: {acc*100:.2f}%')
                avg_acc += acc
            avg_acc /= 10
            print(f'Average acc: {avg_acc*100:.2f}%')
        return avg_acc

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, avg_acc):
        torch.save(
            {
                "state_dict": self.netG.state_dict(),
                "lr": self.args.lr,
                "last_epoch": self.current_epoch,
                "avg_acc": avg_acc,
            },
            f"{self.args.outf_checkpoint}/netG_epoch_{self.current_epoch}_{avg_acc}.pth",
        )
        torch.save(
            {
                "state_dict": self.netD.state_dict(),
                "lr": self.args.lr,
                "last_epoch": self.current_epoch,
                "avg_acc": avg_acc,
            },
            f"{self.args.outf_checkpoint}/netD_epoch_{self.current_epoch}_{avg_acc}.pth",
        )
        print(f"save ckpt")


if __name__ == "__main__":
    sys.argv = ["training.py", "--save_root", "./data"]

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--dvc', type=str, default='cuda:0', help='running device: cuda or cpu')
    parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
    parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
    parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')
    parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
    parser.add_argument('--update_every', type=int, default=50, help='training frequency')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
    parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
    parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
    parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
    parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')
    
    opt = parser.parse_args()
    opt.dvc = torch.device(opt.dvc)
    agent = DQN_agent(**vars(opt))

    cudnn.benchmark = True
    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.outf, exist_ok=True)

    train_dqn = TrainDQN(args)

    if args.resume:
        testnetDQN = torch.load(args.netDQN)
    else:
        wandb.init(
            project = 'Engagement',
            config = {
                    "batch_size":args.batch_size, 
                    "epoch": 60, 
                    "embedding": "nn.Linear", 
                    "Type": "Test111",
                    "Block_size": 'bigger',
                    "Resume": False
            },
            name = "Test1"
        )
        train_dcgan.train(train_dataloader, test_dataloader)
        wandb.finish()
    
