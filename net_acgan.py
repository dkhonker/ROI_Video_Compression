import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import math
# from detect import creat_weight_map, cal_roi_mseloss, cal_roi_ploss
from subnet import *
import torchac

def save_model(G_model, D_model, G_opt, D_opt, iter):
    state = {
        'G_model': G_model.state_dict(),
        'D_model': D_model.state_dict(),
        'G_optimizer': G_opt.state_dict(),
        'D_optimizer': D_opt.state_dict()
    }
    # torch.save(state, "./snapshot/iter{}.model".format(iter))
    torch.save(state, "/data1/qiuzd/snapshot/iter{}.model".format(iter))

def load_model(G_model, D_model, G_opt, D_opt, f):
    with open(f, 'rb') as f:
        checkpoint = torch.load(f)
        G_dict = checkpoint
        G_model_dict = G_model.state_dict()
        G_dict = {k: v for k, v in G_dict.items() if k in G_model_dict}
        G_model_dict.update(G_dict)
        G_model.load_state_dict(G_model_dict)
        # checkpoint = torch.load(f)
        # G_dict = checkpoint['G_model']
        # G_model_dict = G_model.state_dict()
        # G_dict = {k: v for k, v in G_dict.items() if k in G_model_dict}
        # G_model_dict.update(G_dict)
        # G_model.load_state_dict(G_model_dict)
        # D_dict = checkpoint['D_model']
        # D_model_dict = D_model.state_dict()
        # D_dict = {k: v for k, v in D_dict.items() if k in D_model_dict}
        # D_model_dict.update(D_dict)
        # D_model.load_state_dict(D_model_dict)
        # G_opt.load_state_dict(checkpoint['G_optimizer'])
        # D_opt.load_state_dict(checkpoint['D_optimizer'])
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0


class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        # self.imageCompressor = ImageCompressor()
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.Q = None
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)
        # self.bitEstimator_feature = BitEstimator(out_channel_M)
        self.embed = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Sigmoid()
        )
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe, weight_map, weighted_img,
                quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None):
        estmv = self.opticFlow(input_image, referframe)
        # torch.Size([4, 2, 256, 256])
        mvfeature = self.mvEncoder(estmv)
        # torch.Size([4, 128, 16, 16])
        mvfeature *= self.embed(weight_map)
        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)
        quant_mv_upsample = self.mvDecoder(quant_mv)
        prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

        input_residual = (input_image - prediction) * (weight_map == 1)

        feature = self.resEncoder(input_residual)
        batch_size = feature.size()[0]

        z = self.respriorEncoder(feature)

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.respriorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        recon_res = self.resDecoder(compressed_feature_renorm)
        recon_image = prediction + recon_res

        clipped_recon_image = recon_image.clamp(0., 1.)


# distortion
        # warploss = torch.mean((warpframe - weighted_img).pow(2))
        # interloss = torch.mean((prediction - weighted_img).pow(2))
        # mse_loss = torch.mean((recon_image - input_image).pow(2))
        # print(torch.sum((recon_image - input_image).pow(2) * (weight_map == 1)), torch.sum(weight_map == 1))
        warploss = torch.mean(((warpframe - input_image) * weight_map).pow(2))
        interloss = torch.mean(((prediction - input_image) * weight_map).pow(2))
        mse_loss = torch.sum((recon_image - input_image).pow(2) * (weight_map == 1)) / torch.sum(weight_map == 1)
# bit per pixel
        def feature_probs_based_sigma(feature, sigma):
            
            def getrealbitsg(x, gaussian):
                # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(gaussian.cdf(i - 0.5).view(n,c,h,w,1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits


            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            
            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbitsg(feature, gaussian)
                total_bits = real_bits

            return total_bits, probs
        def iclr18_estrate_bits_z(z):
            
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(z)
                total_bits = real_bits

            return total_bits, prob
        def iclr18_estrate_bits_mv(mv):

            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(mv)
                total_bits = real_bits

            return total_bits, prob
        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        # entropy_context = entropy_context_from_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

        im_shape = input_image.size()

        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv
        
        return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dis = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 3, 1, 1)),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 4, 2, 1)),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, 3, 1, 1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 128, 4, 2, 1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, 3, 1, 1)),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, 4, 2, 1)),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.last = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cls_map = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 1, 2, 2),
            # nn.Sigmoid()
        )
        self.cls_logit = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                   nn.Conv2d(512, 1024, kernel_size=1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(1024, 1, kernel_size=1))
        # self.cls_logit = nn.Linear(512, 1)

    def forward(self, img):
        out = self.last(self.dis(img))
        pred_map = self.cls_map(out)
        pred_logit = self.cls_logit(out).view(-1, 1)
        return pred_map, pred_logit
