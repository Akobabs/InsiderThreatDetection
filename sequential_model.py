import torch
import torch.nn as nn
import os

class SequentialModel(nn.Module):
    """LSTM-based model for sequential data classification."""
    def __init__(self, sequence_length, input_size, hidden_size, output_size, batch_first=True):
        super(SequentialModel, self).__init__()
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = 0.5

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=batch_first, dropout=self.dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size * 2, batch_first=batch_first, dropout=self.dropout_rate)
        self.linear = nn.Linear(hidden_size * 2 * sequence_length, output_size)

    def forward(self, input_data):
        """Forward pass through the LSTM and linear layers."""
        output1, (hn1, cn1) = self.lstm1(input_data)
        output2, (hn2, cn2) = self.lstm2(output1)
        output2 = output2.reshape(output2.shape[0], -1)
        output = self.linear(output2)
        return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SequentialModel(
        sequence_length=100,
        input_size=128,
        hidden_size=256,
        output_size=5,
        batch_first=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.NLLLoss().to(device)

    input_data = torch.randn(5, 100, 128).to(device)
    labels = torch.tensor([1, 0, 2, 3, 4], dtype=torch.long).to(device)

    print("Training model...")
    for epoch in range(10):
        output = model(input_data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.5f}")

    print(f"Output shape: {output.shape}")