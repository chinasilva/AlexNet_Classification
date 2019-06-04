
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
from src.AlexNet import AlexNet

def main():

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	epochs = 100
	batch_size = 64

	net = AlexNet(10).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), weight_decay = 0, amsgrad = False, lr = 0.001, betas = (0.9, 0.999), eps = 1e-08)

	transform_train = transforms.Compose([
	        transforms.Resize(32),
	        transforms.CenterCrop(32),
	        transforms.ToTensor(),
	        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	    ])
	dataset = datasets.CIFAR10("datasets/", train=True, download=True,transform=transform_train)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	transform_test = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	    ])
	testdataset = datasets.CIFAR10("datasets/", train=False, download=True,transform=transform_test)
	testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False)



	losses = []
	for i in range(epochs):
		# net.train()
		since = time.time()
		print("epochs: {}".format(i))
		for j, (input, target) in enumerate(dataloader):
			input, target = input.to(device), target.to(device)
			# print("target",target.size())
			# print("input",input.size())
			output = net(input)
			# print("output",output.size())
			loss = criterion(output, target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if j % 10 == 0:
				losses.append(loss.float())
				print("[epochs - {0} - {1}/{2}]loss: {3}".format(i, j, len(dataloader), loss.float()))
		time_elapsed = time.time() - since
		print('time_cost:{}'.format(time_elapsed))

		with torch.no_grad():
			net.eval()
			correct = 0.
			total = 0.
			for input, target in testdataloader:
				input, target = input.to(device), target.to(device)
				output = net(input)
				_, predicted = torch.max(output.data, 1)
				total += target.size(0)
				correct += (predicted == target).sum()
				accuracy = correct.float() / total
			print("[epochs - {0}]Accuracy:{1}%".format(i + 1, (100 * accuracy)))
		# torch.save(net, "models/net.pth")
	torch.save(model_object.state_dict(), 'AlexNet_Classification/models/net.pkl')
	# model_object.load_state_dict(torch.load('params.pkl'))

