import torch
import soundfile as sf
import numpy as np
import random
import h5py
from torch.utils.data import Dataset, DataLoader


class demanddataset(Dataset):
    '''
    training dataset
    '''

    def __init__(self, file_path, nsamples=16000):
        self.nsamples = nsamples
        # file path is the naame list file of the train dataset
        with open(file_path, 'r') as train_file_list:
            self.file_list = [
                line.strip() for line in train_file_list.readlines()
            ]

            self.nsamples = nsamples

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')
        feature = reader['noisy_raw'][:] # type:ignore
        label = reader['clean_raw'][:] # type:ignore
        reader.close()

        feature = torch.from_numpy(feature).float()  #[len]
        label = torch.from_numpy(label).float()

        size1 = feature.size(-1)
        size2 = label.size(-1)

        assert size1 == size2

        if size1 < self.nsamples:
            units = self.nsamples // size1
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(label) # 重复整数次倍
                noisy_ds_final.append(feature)
            clean_ds_final.append(label[:self.nsamples%size2]) # 余下使用语音继续补全
            noisy_ds_final.append(feature[:self.nsamples%size1])
            label = torch.cat(clean_ds_final,dim=-1)
            feature = torch.cat(noisy_ds_final,dim=-1)
        else:
            wav_start = random.randint(0,size1-self.nsamples)
            feature = feature[wav_start:wav_start+self.nsamples]
            label = label[wav_start:wav_start+self.nsamples]

        return feature,label # [b,len]

def load(file_path,batch_size,nsamples):
    data = demanddataset(file_path,nsamples)
    loader = DataLoader(dataset=data,batch_size=batch_size,shuffle=True,drop_last=True)
    return loader

if __name__ == '__main__':
    # train_set = demanddataset('../vb28_train_list',nsamples=48000)
    # train_loader = DataLoader(train_set,batch_size=2,shuffle=False)
    # for k,(f,l) in enumerate(train_loader):
    #     print(l.size())
    val_set = demanddataset('../vb28_test_list',nsamples=32000)
    val_loader = DataLoader(val_set,batch_size=4,shuffle=False,drop_last=True)
    for k,batch in enumerate(val_loader):
        print(batch[1].size())
