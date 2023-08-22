import os
import argparse
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm
from src.model import Model_big, Model_attn, Model_ori, Model_cbigger, Model_pbigger, Model_pbiggerc, Model_connect
from src.utils import *
from src.compute_metrics import compute_metrics


@torch.no_grad()
def enhanced_one_track(model,
                       audio_path,
                       saved_dir,
                       cut_len,
                       n_fft=512,
                       hop=256):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)  # type:ignore
    assert sr == 16000
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1)).cuda()
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    print('audio len', length)
    # frame_num = int(np.ceil(length / 100))
    # padded_len = frame_num * 100
    # padding_len = padded_len - length
    padded_len = length + 2 * 384
    noisy = torch.cat([noisy[:, :384], noisy, noisy[:, -384:]], dim=-1).cuda()
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))
    print('noisy', noisy.size())
    noisy_spec = torch.stft(noisy,
                            n_fft,
                            hop,
                            window=torch.hamming_window(n_fft).cuda(),
                            onesided=True,
                            return_complex=True)
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)  # [b,c,t,f]
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3,
                                          2), est_imag.permute(0, 1, 3,
                                                               2)  #[b,1,t,f]
    est_spec_un = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(est_spec_un,
                            n_fft,
                            hop,
                            window=torch.hamming_window(n_fft).cuda(),
                            onesided=True)
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[384:384 + length].cpu().numpy()
    print('est len', len(est_audio))
    assert len(est_audio) == length
    save_path = os.path.join(saved_dir, name)
    sf.write(save_path, est_audio, sr)  # type:ignore

    return est_audio, length


def main(args):
    print(args.model_type)

    if 'big' == args.model_type:
        model = Model_big(num_channel=args.num_channel,
                          num_features=args.n_fft // 2 + 1).cuda()
    elif 'attn' == args.model_type:
        model = Model_attn(num_channel=args.num_channel,
                           num_features=args.n_fft // 2 + 1).cuda()

    elif 'ori' == args.model_type:
        model = Model_ori(num_channel=args.num_channel,
                          num_features=args.n_fft // 2 + 1).cuda()

    elif 'bigger1' == args.model_type:
        model = Model_cbigger(num_channel=args.num_channel,
                              num_features=args.n_fft // 2 + 1).cuda()

    elif 'bigger2' == args.model_type:
        model = Model_pbigger(num_channel=args.num_channel,
                              num_features=args.n_fft // 2 + 1).cuda()

    elif 'bigger3' == args.model_type:
        model = Model_pbiggerc(num_channel=args.num_channel,
                               num_features=args.n_fft // 2 + 1).cuda()
    elif 'connect' == args.model_type:
        model = Model_connect(num_channel=args.num_channel,
                              num_features=args.n_fft // 2 + 1).cuda()

    checkpoint = torch.load(f'./model/{args.model_name}')
    model.load_state_dict(checkpoint['state_dict'])  # type:ignore

    model.eval()  # type:ignore

    with torch.no_grad():
        output_path = './enhanced'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    audio_list = os.listdir(args.noisy_dir)
    num = len(audio_list)
    metrics_total = np.zeros(6)
    for audio in tqdm(audio_list):
        noisy_path = os.path.join(args.noisy_dir, audio)
        clean_path = os.path.join(args.clean_dir, audio)
        est_audio, length = enhanced_one_track(
            model,  # type:ignore
            noisy_path,
            output_path,
            16000 * 16,
            args.n_fft,
            args.n_fft // 2)
        clean_audio, sr = sf.read(clean_path)
        assert sr == 16000
        metrics = compute_metrics(clean_audio, est_audio, sr, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ',
          metrics_avg[2], 'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4],
          'stoi: ', metrics_avg[5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_channel', type=int, default=64)
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--hop', type=int, default=256)
    parser.add_argument(
        '--model_type',
        type=str,
        default='big',
        help='model type: ori,attn,big,bigger1,bigger2,bigger3')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--noisy_dir',
                        type=str,
                        default='F:/dataset/VB28_16/noisy_test',
                        help='noisy wav')
    parser.add_argument('--clean_dir',
                        type=str,
                        default='F:/dataset/VB28_16/clean_test',
                        help='clean wav')
    args = parser.parse_args()
    main(args)
