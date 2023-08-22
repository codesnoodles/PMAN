import os
import argparse
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
from src.model import Model_attn, Model_big, Model_ori, Model_cbigger, Model_pbigger, Model_pbiggerc, Model_connect
from src.utils import *
from src.dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",
                    type=int,
                    default=160,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--log_interval", type=int, default=1000)
parser.add_argument("--decay_epoch",
                    type=int,
                    default=40,
                    help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="lr initial")
parser.add_argument("--cut_len", type=int, default=32000, help="wav's length")
parser.add_argument("--loss_weights",
                    type=list,
                    default=[0.7, 0.2, 0.1, 1],
                    help="mag,ri,ph,time")
args = parser.parse_args()
logger = get_logger('./exp.log')


class Trainer:

    def __init__(self, train_ds, test_ds):
        seed_init()
        self.n_fft = 512
        self.hop = 256
        self.train_ds = train_ds
        self.test_ds = test_ds
        # self.model = Model_big(num_channel=64,
        #                        num_features=self.n_fft // 2 + 1).cuda()
        # self.model = Model_pbigger(num_channel=64,
        #                            num_features=self.n_fft // 2 + 1).cuda()
        # self.model = Model_cbigger(num_channel=64,
        #                            num_features=self.n_fft // 2 + 1).cuda()
        self.model = Model_pbiggerc(num_channel=64,
                                    num_features=self.n_fft // 2 + 1).cuda()
        # self.model = Model_connect(num_channel=64,
        #                            num_features=self.n_fft // 2 + 1).cuda()
        summary(self.model, [(2, 2, 251, 257)])
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=args.init_lr)

    def train_step(self, dataset):
        clean = dataset[1].cuda()
        noisy = dataset[0].cuda()

        # normalization
        clean, noisy = Normalization(clean, noisy)

        self.optimizer.zero_grad()

        # stft
        noisy_spec = torch.stft(noisy,
                                self.n_fft,
                                self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True,
                                return_complex=True)
        clean_spec = torch.stft(clean,
                                self.n_fft,
                                self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True,
                                return_complex=True)

        # compress
        clean_spec = power_compress(clean_spec)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3,
                                                        2)  # [b,c,t,f]

        # real imag
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        # model out
        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(
            0, 1, 3, 2)  #[b,1,t,f]
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
        est_phase = torch.angle(torch.complex(est_real, est_imag))
        clean_phase = torch.angle(torch.complex(clean_real, clean_imag))

        # spec-->time
        est_spec_un = power_uncompress(est_real,
                                       est_imag).squeeze(1)  # [b,f,t]
        est_audio = torch.istft(est_spec_un,
                                self.n_fft,
                                self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        length = est_audio.size(-1)
        clean_spec_un = power_uncompress(clean_real,
                                         clean_imag).squeeze(1)  # [b,f,t]
        clean_audio = torch.istft(clean_spec_un,
                                  self.n_fft,
                                  self.hop,
                                  window=torch.hamming_window(
                                      self.n_fft).cuda(),
                                  onesided=True)

        # loss
        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(
            est_imag, clean_imag)
        loss_pahse = F.mse_loss(est_phase, clean_phase)
        loss_time = torch.mean(torch.abs(est_audio - clean_audio))

        loss = args.loss_weights[0] * loss_mag + args.loss_weights[
            1] * loss_ri + args.loss_weights[
                2] * loss_pahse + args.loss_weights[3] * loss_time
        loss.backward()
        self.optimizer.step()

        # # pesq calculate
        # est_audio_list = list(est_audio.detach().cpu().numpy())
        # clean_audio_list = list(clean_audio.cpu().numpy()[:,:length])

        return loss.item()

    @torch.no_grad()
    def test_step(self, dataset):
        clean = dataset[1].cuda()
        noisy = dataset[0].cuda()
        clean, noisy = Normalization(clean, noisy)

        noisy_spec = torch.stft(noisy,
                                self.n_fft,
                                self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True,
                                return_complex=True)
        clean_spec = torch.stft(clean,
                                self.n_fft,
                                self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True,
                                return_complex=True)
        # compress
        clean_spec = power_compress(clean_spec)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3,
                                                        2)  # [b,c,t,f]
        # real imag
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        # model out
        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(
            0, 1, 3, 2)  #[b,1,t,f]
        # mag phase
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
        est_phase = torch.angle(torch.complex(est_real, est_imag))
        clean_phase = torch.angle(torch.complex(clean_real, clean_imag))
        # spec-->time
        est_spec_un = power_uncompress(est_real,
                                       est_imag).squeeze(1)  # [b,f,t]
        est_audio = torch.istft(est_spec_un,
                                self.n_fft,
                                self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        length = est_audio.size(-1)
        clean_spec_un = power_uncompress(clean_real,
                                         clean_imag).squeeze(1)  # [b,f,t]
        clean_audio = torch.istft(clean_spec_un,
                                  self.n_fft,
                                  self.hop,
                                  window=torch.hamming_window(
                                      self.n_fft).cuda(),
                                  onesided=True)
        # loss
        loss_mag = F.mse_loss(est_mag, clean_mag)
        loss_ri = F.mse_loss(est_real, clean_real) + F.mse_loss(
            est_imag, clean_imag)
        loss_pahse = F.mse_loss(est_phase, clean_phase)
        loss_time = torch.mean(torch.abs(est_audio - clean_audio))

        loss = args.loss_weights[0] * loss_mag + args.loss_weights[
            1] * loss_ri + args.loss_weights[
                2] * loss_pahse + args.loss_weights[3] * loss_time

        # pesq calculate
        est_audio_list = est_audio.detach().cpu()
        clean_audio_list = clean_audio.cpu()

        batch_pesq, batch_stoi = get_scores(clean_audio_list, est_audio_list)
        batch_pesq = batch_pesq // args.batch_size
        batch_stoi = batch_stoi // args.batch_size

        return loss.item(), batch_pesq, batch_stoi

    def test(self):
        self.model.eval()
        loss_total = 0.
        pesq_total = 0.
        stoi_total = 0.
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, b_pesq, b_stoi = self.test_step(batch)
            loss_total += loss
            pesq_total += b_pesq
            stoi_total += b_stoi
        loss_avg = loss_total / step  # type:ignore
        pesq_avg = pesq_total / step  # type:ignore
        stoi_avg = stoi_total / step  # type:ignore

        template = 'test loss:{},pesq:{},stoi:{}'
        logger.info(template.format(loss_avg, pesq_avg, stoi_avg))

        return loss_avg, pesq_avg, stoi_avg

    def train(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=args.decay_epoch,
                                                    gamma=0.5)

        for epoch in range(args.epochs):
            self.model.train()
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                total_len = len(self.train_ds)
                loss = self.train_step(batch)
                template = 'epoch:{},step/total:{}/{},loss:{}'
                print(template.format(epoch, step, total_len, loss))
                if (step % args.log_interval) == 0:
                    logger.info(template.format(epoch, step, total_len, loss))
            test_loss, pesq_avg, stoi_avg = self.test()
            test_template = 'epoch:{},loss:{},pesq:{},stoi:{}'
            logger.info(
                test_template.format(epoch, test_loss, pesq_avg, stoi_avg))
            path = os.path.join(
                './model', 'epoch_' + str(epoch) + '_' + str(test_loss)[:3])
            if not os.path.exists('./model'):
                os.makedirs('./model')
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(checkpoint, path)
            scheduler.step()


def main():
    train_ds = load('./vb28_train_list', args.batch_size, args.cut_len)
    test_ds = load('./vb28_test_list', args.batch_size, args.cut_len)
    trainer = Trainer(train_ds, test_ds)
    trainer.train()


if __name__ == '__main__':
    main()
