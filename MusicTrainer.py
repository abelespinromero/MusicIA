import torch
import torch.nn as nn
import torch.optim as optim

class MusicTrainer:
    def __init__(self, model, data, batch_size=32, learning_rate=0.001, num_epochs=10):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            hidden = self.model.init_hidden(self.batch_size)
            for i in range(0, len(self.data)-self.batch_size, self.batch_size):
                inputs = self.data[i:i+self.batch_size]
                targets = self.data[i+1:i+self.batch_size+1]
                
                inputs = torch.from_numpy(inputs).float()
                targets = torch.from_numpy(targets).float()
                
                self.optimizer.zero_grad()
                hidden = tuple(h.detach() for h in hidden)
                outputs, hidden = self.model(inputs, hidden)
                
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                if i % 1000 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i}/{len(self.data)-self.batch_size}], Loss: {loss.item()}')

