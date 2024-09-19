from utils.config import set_config
from utils.utils import CustomDataset
from utils.net import NET
from utils.trainer import Trainer

import torch, torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split

### set seeds
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

show_net_architecture = False
config = set_config()

print(f"Dataset path {config['path']}")
print(f"Dataset temps ranged {config['sample_bounds']} normalized to [0,1]")

T = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize(config['img_shape'][1:]),
	])

dataset = CustomDataset(samples=config['samples'], transforms=T)
train_dataset, test_dataset = random_split(  dataset, [config['split'], 1 - config['split']]  )
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], drop_last=True, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], drop_last=True, shuffle=True)
net = NET().to(config['device'])
if show_net_architecture:
	print(net)
trainer = Trainer(net, config)
trainer.train(train_dataloader, save=True, load=True, show_graphs=True)
trainer.test(test_dataloader)





