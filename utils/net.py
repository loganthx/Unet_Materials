import torch, torch.nn as nn



class NET(nn.Module):
	def __init__(self):
		super().__init__()
		# self.drop = nn.Dropout(p=0.5)
		self.relu = nn.GELU()
		# self.relu = nn.ReLU()
		# self.relu = nn.LeakyReLU(0.2)
		self.sigmoid = nn.Sigmoid()
		# IN: (b, 3, 64, 64)
		self.conv_0 = nn.Conv2d(3, 32, 3, padding=1) # (b, 32, 64, 64)
		self.conv_1 = nn.Conv2d(32, 64, 2, stride=2) # (b, 64, 32, 32)
		self.conv_2 = nn.Conv2d(64, 128, 2, stride=2) # (b, 128, 16, 16)
		self.conv_3 = nn.Conv2d(128, 256, 2, stride=2) # (b, 256, 8, 8)
		self.transp_3 = nn.ConvTranspose2d(256, 256, 3, padding=1) # (b, 256, 8, 8)
		# cat(out_conv_3, out_transp_3)
		self.transp_2 = nn.ConvTranspose2d(256+256, 128, 2, stride=2) # (b, 128, 16, 16)
		# cat(out_conv_2, out_transp_2)
		self.transp_1  =  nn.ConvTranspose2d(128+128, 64, 2, stride=2) # (b, 64, 32, 32)
		# cat(out_conv_1, out_transp_1)
		self.transp_0  =  nn.ConvTranspose2d(64+64, 32, 2, stride=2) # (b, 32, 64, 64)
		# out_transp_0
		self.final_transp =  nn.ConvTranspose2d(32, 1, 1) # (b, 1, 64, 64)
		# OUT:(b, 1, 64, 64)
		self.fc1 = nn.Linear(1 * 64 * 64, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 1)
		# OUT:(b, 1)


	def forward(self, x):	
		inp = x 
		out_conv_0 = self.relu(self.conv_0(inp))
		out_conv_1 = self.relu(self.conv_1(out_conv_0))
		out_conv_2 = self.relu(self.conv_2(out_conv_1))
		out_conv_3 = self.relu(self.conv_3(out_conv_2)) 
		out_transp_3 = self.relu(self.transp_3(out_conv_3))
		cat_3 = torch.cat((out_conv_3, out_transp_3), dim=1)
		out_transp_2 = self.relu(self.transp_2(cat_3))
		cat_2 = torch.cat((out_conv_2, out_transp_2), dim=1)
		out_transp_1 = self.relu(self.transp_1(cat_2))
		cat_1 = torch.cat((out_conv_1, out_transp_1), dim=1)
		out_transp_0 = self.relu(self.transp_0(cat_1))

		conv_out = self.sigmoid( self.final_transp(out_transp_0) )
		conv_out_flat = conv_out.view(-1, 1 * 64 * 64)
		fc1 = self.relu(self.fc1(conv_out_flat)) 
		fc2 = self.relu(self.fc2(fc1)) # (b, 64) = (b, 1, 8, 8)
		flat_out = fc2.view(-1, 1, 8, 8)
		fc3 = self.fc3(fc2)
		out = self.sigmoid(fc3)

		return out, conv_out, self.sigmoid(flat_out)














