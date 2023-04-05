import torch

from model_tracing import Model



device = 'cpu'
model = Model(embedding_dim = 128, hidden_dim = 1000)
model.load_state_dict(torch.load('./saved_model/best_model.pt'))
model.eval()
model.to(device)


example = torch.randint(1, 10, (1,3)).to(device)


state_h, state_c = model.init_state(3)
model_scripted = torch.jit.trace(model,(example, state_h.to(device), state_c.to(device)))
# model_scripted = torch.jit.trace(model,(example))

model_scripted.save("./saved_model/cpu_model_tup2.pt")
