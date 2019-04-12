import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

CORPUS_FILE = 'tiny_corpus.txt'
NDIM = 100
CONTEXT_SIZE = 3
DEVICE = 'cpu'
BATCH_SIZE = 64

with open(CORPUS_FILE, 'r') as f:
    corpus = f.read().split()

vocab = set(corpus)
word2idx = {word: i for i, word in enumerate(vocab)}
trigrams = [([corpus[i], corpus[i + 1]], corpus[i + 2])
            for i in range(len(corpus) - 2)]
train_data = [[word2idx[a], word2idx[b]] for (a,b),_ in trigrams]
target_data = [word2idx[w] for _,w in trigrams]

class Word2Vec(nn.Module):
    def __init__(self, vocab_size:int, ndim:int=100, context_size:int=5):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, ndim)
        self.linear = nn.Linear(ndim, vocab_size, False)

    def forward(self, x):
        vector = self.embeddings(x)
        out = self.linear(vector.sum(dim=1))
        return out

losses = []
loss_function = nn.CrossEntropyLoss()
model = Word2Vec(len(vocab), NDIM, CONTEXT_SIZE).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=1e-3)

train_data = torch.LongTensor(train_data)
target_data = torch.LongTensor(target_data)

for epoch in range(10):
    total_loss = 0
    for start in tqdm.tqdm(range(0, len(trigrams), BATCH_SIZE)):
        end = min(len(trigrams), start + BATCH_SIZE)

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        x = train_data[start:end].to(DEVICE)
        y = target_data[start:end].to(DEVICE)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(x)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, y)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    print(total_loss)
    losses.append(total_loss)

print(losses)  # The loss decreased every iteration over the training data!
