import torch
from torch import nn
from torch.nn import functional as F
import math  # 导入math模块

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_seq_length = max_seq_length
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_embedding = self.embedding(src) * math.sqrt(self.d_model)
        tgt_embedding = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        output = self.decoder(tgt_embedding, src_embedding, tgt_mask=tgt_mask, memory_mask=src_mask)
        output = self.output(output)
        return output
        
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, vocab=None, max_seq_length=128):
        if vocab is None:
            self.vocab = {'<GUOXIUZHI>': 0, '<BOS>': 1, '<EOS>': 2}
            self.build_vocab(texts)
        else:
            self.vocab = vocab
        
        self.max_seq_length = max_seq_length
        self.data = [self.encode_text(text) for text in texts]
    
    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, _ in counter.most_common():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
    
    def encode_text(self, text):
        tokens = ['<BOS>'] + text.split() + ['<EOS>']
        token_ids = [self.vocab[token] for token in tokens if token in self.vocab]
        padding = [0] * (self.max_seq_length - len(token_ids))
        return torch.tensor(token_ids + padding, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 示例数据
texts = ["这是一个例子", "这是第二个例子","这是一个郭秀志的例子","这是一个郭的例子","这是一个郭的例子","这是一个秀志的例子","这是一个guo的例子"]
dataset = TextDataset(texts)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerDecoder(
    vocab_size=len(dataset.vocab),
    d_model=8,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    max_seq_length=dataset.max_seq_length
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)

def train(model, dataloader, criterion, optimizer, epochs=28):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch[:-1], batch[1:])
            loss = criterion(output.view(-1, output.shape[-1]), batch[1:].view(-1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

train(model, dataloader, criterion, optimizer)

def predict(model, dataset, start_text, max_length=50, temperature=0.1):
    model.eval()
    with torch.no_grad():
        input_ids = dataset.encode_text(start_text).unsqueeze(1).to(device)
        for _ in range(max_length):
            output = model(input_ids[:-1], input_ids[1:])
            next_token_probs = output[-1].squeeze() / temperature
            next_token_probs = F.softmax(next_token_probs, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, 1).item()
            if next_token_id == dataset.vocab['<EOS>']:
                break
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=0)
        return ' '.join([list(dataset.vocab.keys())[i] for i in input_ids.squeeze().tolist()[1:-1]])

start_text = "这是"
generated_text = predict(model, dataset, start_text)
print(generated_text)
