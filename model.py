import torch
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class DataHelper():
    def __init__(self,embedding_path):
        self.id_to_token = []
        self.token_to_id = {}
        # Create word embeddings and initialise
        self.id_to_token = []
        self.token_to_id = {}
        self.pretrained = []
        for line in open(embedding_path):
            parts = line.strip().split()
            word = parts[0].lower()
            vector = [float(v) for v in parts[1:]]
            self.token_to_id[word] = len(self.id_to_token)
            self.id_to_token.append(word)
            self.pretrained.append(vector)
        self.NWORDS = len(self.id_to_token)
        self.DIM_WORDS = len(self.pretrained[0])

    def sent2index(self,sent,max_seq = 32):
        index = [0 for i in range(max_seq)]
        unk_id = self.token_to_id.get('<unka>',0)
        for i,word in enumerate(sent):
            if i >= max_seq:
                break
            id = self.token_to_id.get(word,unk_id)
            index[i] = id
        return index

class EsimEmbedder(nn.Module):
    def __init__(self, embeds_dim,hidden_size,num_word,weight_matrix):
        super(EsimEmbedder, self).__init__()
        self.dropout = 0.2
        linear_size = 128
        self.hidden_size = hidden_size
        self.embeds_dim = embeds_dim
        weight = torch.FloatTensor(weight_matrix).cuda()
        self.embeds = nn.Embedding.from_pretrained(weight)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.GRU(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=False)
        self.lstm2 = nn.GRU(self.hidden_size*4, self.hidden_size, batch_first=True, bidirectional=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Sequential(
                nn.BatchNorm1d(self.hidden_size * 4),
                nn.Linear(self.hidden_size * 4, linear_size),
                nn.ELU(inplace=True),
                nn.BatchNorm1d(linear_size),
                nn.Dropout(self.dropout),
                nn.Linear(linear_size, linear_size),
                nn.ELU(inplace=True),
                nn.BatchNorm1d(linear_size),
                nn.Dropout(self.dropout),
            )
    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention , dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) , dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, input):
        # print(input)
        # batch_size * seq_len
        sent1, sent2 = torch.tensor(input[0]).cuda(), torch.tensor(input[1]).cuda()
        sent1.to(self.device)
        sent2.to(self.device)
        mask1, mask2 = sent1.eq(0), sent2.eq(0)
        
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        
        

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        
        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        

        sen1 = torch.mean(q1_align,1)
        sen2 = torch.mean(q2_align,1)
        # print(q1_align.shape)
        # return torch.cat([sen1,sen2],-1)
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # feature_vectors
        x = torch.cat([q1_rep, q2_rep], -1)
        x = self.fc(x)
        return x


class PyTorchModel(torch.nn.Module):
    def __init__(self,FEATURES,HIDDEN):
        super().__init__()
        self.data_helper = DataHelper("glove-ubuntu.txt")
        self.esim_embedder = EsimEmbedder(hidden_size=128,embeds_dim=self.data_helper.DIM_WORDS,num_word = self.data_helper.NWORDS,weight_matrix=self.data_helper.pretrained)
        feature_size = FEATURES+128
        self.hidden1 = torch.nn.Linear(feature_size, HIDDEN)
        self.nonlin1 = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.nonlin2 = torch.nn.ReLU()
        self.norm = torch.nn.Softmax(dim=-1)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def get_ids(self, words):
        return self.data_helper.sent2index(words,max_seq=64)

    def forward(self, query, options, gold, lengths, query_no):
        answer = max(gold)
        #print(gold)
        label = [0 for i in range(len(options))]
        label[answer] = 1
        # Concatenate the other features
        features = torch.tensor([v[1] for v in options]).cuda()
        features.to(self.device)
        opt_tok = [v[0] for v in options]
        query_tok =  [query for v in range(len(options))]
        sent_feature = self.esim_embedder([query_tok,opt_tok])
        final_features = torch.cat([features,sent_feature],-1)
        # print(final_features.shape)
        h1 = self.nonlin1(self.hidden1(final_features))
        h2 = self.nonlin2(self.hidden2(h1))
        h3 = self.norm(h2)
        #print(h2.shape)
        scores = torch.sum(h2, 1)
        #print(scores.shape)
        output_scores = torch.unsqueeze(torch.unsqueeze(scores, 0), 2)

        # Get loss and prediction
        true_out = torch.tensor([[answer]]).cuda()
        #print(output_scores.shape)
        loss = self.loss_function(output_scores, true_out)
        predicted_link = torch.argmax(output_scores, 1)
        return loss, predicted_link