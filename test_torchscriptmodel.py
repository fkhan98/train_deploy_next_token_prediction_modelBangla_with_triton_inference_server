import torch
import io
from model import Model



example = torch.randint(3, 100, (1,3))
print(example)
model = Model(embedding_dim = 128, hidden_dim = 1000)
state_h, state_c = model.init_state(3)

torchscript_model = torch.jit.load('./saved_model/cpu_model6.pt')

torchscript_model.eval()
print(torchscript_model(example).shape)