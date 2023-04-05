import torch
#from torch import nn

class Model(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size = 5000, num_layers = 2):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm = torch.nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers)

        self.leakyrelu = torch.nn.LeakyReLU(0.1)

        self.dropout = torch.nn.Dropout(p=0.2)

        self.fc = torch.nn.Linear(self.hidden_dim, self.vocab_size)

        self.softmax = torch.nn.Softmax() 



    def forward(self, x, hidden_state, cell_state):

        embed = self.embedding(x)
        prev_state = (hidden_state, cell_state)
        output, state = self.lstm(embed.to(self.device), prev_state)
        output = self.leakyrelu(output.to(self.device))
        output = self.dropout(output.to(self.device))
        logits = self.fc(output[:, -1, :].to(self.device))
        logits = self.softmax(logits.to(self.device))
        hidden_state = state[0].to(self.device)
        cell_state = state[1].to(self.device)
        return logits.to(self.device), hidden_state, cell_state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.hidden_dim),
                torch.zeros(self.num_layers, sequence_length, self.hidden_dim))