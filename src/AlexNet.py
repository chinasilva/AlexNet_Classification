
import torch
import torch.nn as nn

class AlexNet(nn.Module):

	def __init__(self,num_classes = 10):
		super().__init__()
		self.layer1=nn.Sequential(
			nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
			nn.BatchNorm2d(num_features=96),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2),
		)
		self.layer2=nn.Sequential(
			nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,groups=2, padding=2),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3,stride=2),
		)
		self.layer3=nn.Sequential(
			nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
			nn.BatchNorm2d(num_features=384),
			nn.ReLU(inplace=True),
		)
		self.layer4=nn.Sequential(
			nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
			nn.BatchNorm2d(num_features=384),
			nn.ReLU(inplace=True),
		)
		self.layer5=nn.Sequential(
			nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
			nn.BatchNorm2d(num_features=256),
			nn.ReLU(inplace=True),
		)
		#需要针对上一层改变view
		self.layer6 = nn.Sequential(
            nn.Linear(in_features=256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
		self.layer7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
		self.layer8 = nn.Linear(in_features=4096, out_features= num_classes )


	def forward(self, x_para_1):
		# x = self.layer5(self.layer4(self.layer3(self.layer2())))
		layer1=self.layer1(x_para_1)
		layer2=self.layer2(layer1)
		layer3=self.layer3(layer2)
		layer4=self.layer4(layer3)
		layer5=self.layer5(layer4)
		# print("layer5:",layer5.size())
		x = layer5.view(-1,256)
		# print("x:",x.size())
		# print("layer6:",self.layer6(x).size())
		x = self.layer8(self.layer7(self.layer6(x)))
		return x


