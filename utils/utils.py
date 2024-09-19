from torch.utils.data import Dataset
from PIL import Image

'''
samples = [ [filepath1, temp1], [filepath2, temp2], ... ]
'''

class CustomDataset(Dataset):
	def __init__(self, samples, transforms):
		self.samples = samples 
		self.transforms = transforms 

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		filepath = self.samples[idx][0]
		temp = self.samples[idx][1]
		img = self.transforms(Image.open(filepath))
		return img, temp



