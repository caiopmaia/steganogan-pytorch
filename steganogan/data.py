from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, pasta, crop_size=128):
        
        self.arquivos = sorted(Path(pasta).glob('*.png'))
        if not self.arquivos:
            raise FileNotFoundError(f'Nenhum .png encontrado em {pasta}')


        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, i):
        imagem = Image.open(self.arquivos[i]).convert('RGB')
        x = self.transform(imagem)
        return x * 2 - 1        
