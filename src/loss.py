import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad


class WGANDiscriminatorLoss(nn.Module):
    def __init__(self, penalty_weight, discriminator):
        super(WGANDiscriminatorLoss, self).__init__()
        self.discriminator = discriminator
        self.penalty_weight = penalty_weight

    # Loss function for discriminator
    def forward(self,xreal,xwrong,yreal,ywrong,yfake):
        # Main loss calculation
        wgan_loss = yfake.mean()*0.5 + ywrong.mean()*0.5 - yreal.mean()
        
        # Random linear combination of xreal and xfake
        alpha = Variable(torch.rand(xreal.size(0), 1, 1, 1, out=xreal.data.new()))
        xmix = (alpha * xreal) + ((1. - alpha) * xfake)
        # Run discriminator on the combination
        ymix = self.discriminate(xmix)
        # Calculate gradient of output w.r.t. input
        ysum = ymix.sum()
        grads = grad(ysum, [xmix], create_graph=True)[0]
        gradnorm = torch.sqrt((grads * grads).sum(3).sum(2).sum(1))
        graddiff = gradnorm - 1
        gradpenalty = (graddiff * graddiff).mean() * self.penalty_weight

        # Total loss
        loss = wgan_loss + gradpenalty
        return loss


class WGANGeneratorLoss(nn.BCEWithLogitsLoss):
    # Loss function for generator
    def forward(self, yfake):
        loss = -yfake.mean()
        return loss
