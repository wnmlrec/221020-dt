import torch

class MulNet(torch.nn.Module):
	"""
		Pytorch test model - multiplies input by 0.5
	"""

	def __init__(self):
		super(MulNet, self).__init__()

	def forward(self, x):
		x = torch.mul(x, 0.5)
		return x

def save_test_model():
	traced_net = torch.jit.trace(MulNet(), torch.randn(1, 48000))
	torch.jit.save(traced_net, "models/mul_net.pt")

if __name__ == '__main__':
	save_test_model()