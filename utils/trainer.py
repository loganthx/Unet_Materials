import torch, torch.nn as nn
import numpy as np, os, matplotlib.pyplot as plt

from tqdm import tqdm

class Trainer:
	def __init__(self, net, config):
		self.config = config
		self.device = config['device']
		self.net = net 

	def train(self, dataloader, save=True, load=True, show_graphs=True):
		if load and 'weights.pt' in os.listdir():
				self.net.load_state_dict(torch.load('weights.pt', weights_only=True))

		selected_inp, selected_temp = next(iter(dataloader))
		if show_graphs:
			untrained_outs, untrained_conv_outs, untrained_flat_outs = self.net(selected_inp.to(self.device))

		self.net.train()
		loss_fn = eval(f"nn.{self.config['loss']}()")
		print('loss:', loss_fn)
		
		opt = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'])
		scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.config['lr_step_size'], gamma=self.config['lr_gamma'])
		Min, Max = self.config['sample_bounds']

		for epoch in range(1, self.config['epochs'] + 1):
			epoch_loss=[]

			for n, (inps, temps) in enumerate(tqdm(dataloader)):
				opt.zero_grad()
				temps = temps.unsqueeze(dim=1)
				# Temps = (temps - Min) / (Max - Min)
				# Temps = Temps.float().unsqueeze(dim=1)
				outs, conv_outs, flat_outs = self.net(inps.to(self.device))
				outs_real = (Max - Min)*outs + Min
				loss = loss_fn(outs_real.float(), temps.float().to(self.device))
				loss.backward()
				opt.step()
				epoch_loss.append(loss.item())

			print(f'epoch {epoch} loss {np.mean(epoch_loss)}')
			scheduler.step()


		if show_graphs:
			trained_outs, trained_conv_outs, trained_flat_outs = self.net(selected_inp.to(self.device))
			Inp_0 = selected_inp[0].permute(1,2,0).cpu().numpy()
			Inp_f = selected_inp[0].permute(1,2,0).cpu().numpy()
			epoch_0_conv_outs = untrained_conv_outs[0].permute(1,2,0).detach().cpu().numpy()
			epoch_0_flat_outs = untrained_flat_outs[0].permute(1,2,0).detach().cpu().numpy()
			epoch_f_conv_outs = trained_conv_outs[0].permute(1,2,0).detach().cpu().numpy()
			epoch_f_flat_outs = trained_flat_outs[0].permute(1,2,0).detach().cpu().numpy()
			fig, axs = plt.subplots(2, 3, figsize=(8,8))
			axs[0][0].imshow(Inp_0); axs[0][0].set_title('inps_0')
			axs[0][1].imshow(epoch_0_conv_outs, cmap='gray'); axs[0][1].set_title('convs_0')
			axs[0][2].imshow(epoch_0_flat_outs, cmap='gray'); axs[0][2].set_title('flats_0')
			axs[1][0].imshow(Inp_f); axs[1][0].set_title('inps_f')
			axs[1][1].imshow(epoch_f_conv_outs, cmap='gray'); axs[1][1].set_title('convs_f')
			axs[1][2].imshow(epoch_f_flat_outs, cmap='gray'); axs[1][2].set_title('flats_f')
			for i in range(2): 
				for j in range(3):
					axs[i][j].axis('off')

			# plt.show()
			plt.savefig(f"logs\\plots_{self.config['epochs']}_epochs.png")

		if save:
			torch.save(self.net.state_dict(), 'weights.pt')
			print(f"model weights.pt saved at {os.getcwd()} folder")


	def test(self, dataloader, load=True):
		self.net.eval()

		score = 0 
		total = 0
		Min, Max = self.config['sample_bounds']
		with torch.inference_mode():
			for inps, temps in tqdm(dataloader):
				Temps = (temps - Min) / (Max - Min)
				Temps = Temps.float().unsqueeze(dim=1)
				outs, conv_outs, flat_outs = self.net(inps.to(self.device))
				outs_real = (Max - Min)*outs.detach() + Min
				total += temps.shape[0]
				for i in range(temps.shape[0]):
					if ( outs_real[i].round().int() - temps[i].int() ) == 0:
						score += 1

		config_print = f"CONFIG: \n img_shape {self.config['img_shape']} | batch_size {self.config['batch_size']} | lr {self.config['lr']} | epochs {self.config['epochs']} \n"
		result_print = f"RESULTS: \n accuracy {(score / total) * 100}% | total {total} = {(1 - self.config['split'])*100} % of dataset \n plots saved at logs folder"
		with open('logs\\log.txt', 'w') as f:
			f.write(config_print)
			f.write(result_print)
		print(result_print)




