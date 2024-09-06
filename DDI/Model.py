import torch
import torch.nn as nn
import torch.nn.functional as F

class MHSA(nn.Module):
    def __init__(self,input_dim,num_head):
        super().__init__()
        self.num_head = num_head
        self.input_dim = input_dim

        #Q\K\V
        self.Q = nn.Linear(input_dim,input_dim)
        self.K = nn.Linear(input_dim,input_dim)
        self.V = nn.Linear(input_dim,input_dim)

        # Out
        self.O = nn.Linear(input_dim,input_dim)

    def forward(self, x):
        pass
        # x shape -> [B,input_dim] 不符合MHSA的应用场景
        batch_size = x.shape[0]
        q = self.Q(x).view(batch_size,self.num_head,-1,1)
        k = self.K(x).view(batch_size,self.num_head,-1,1)
        v = self.V(x).view(batch_size,self.num_head,-1,1)

        score = torch.matmul(q,k.transpose(-2,-1))/(self.num_head**0.2)
        score = F.softmax(score,dim=-1)

        out = torch.matmul(score,v)
        out = out.view(batch_size,-1)
        out = self.O(out)
        return out

# input = torch.randn(8,64)
# model = MHSA(64,8)
# pred = model(input)

class GRU_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=32, num_layers=2,batch_first=True,bidirectional=True)

    def forward(self,x):
        batch_size = x.shape[0]
        x = x.view(batch_size,-1,1)
        output, hidden = self.gru(x)
        out = torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim = 1)
        return out
# input = torch.randn(8,64)
# model = GRU_Block()
# pred = model(input)

class TransBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)
        self.MHSA = MHSA(input_dim=64,num_head=8)
        self.BiGRU = GRU_Block()
        self.mlp = nn.Linear(64,64)
    def forward(self,x):
        x = self.ln1(x)
        global_x = self.MHSA(x)
        local_x = self.BiGRU(x)
        x = self.ln2(global_x+local_x)
        x = self.mlp(x)
        return x

class DDI_MODEL(nn.Module):
    def __init__(self):
        super().__init__()

        # multi-modal feature fusion
        ## PCA or MLP
        self.graph = nn.Linear(9,64)
        self.graph_ln = nn.LayerNorm(64)
        self.drug = nn.Linear(25,64)
        self.drug_ln = nn.LayerNorm(64)
        self.smiles = nn.Linear(30000,64)
        self.smiles_ln = nn.LayerNorm(64)
        ##
        self.a1 = nn.Linear(64,64)
        self.a2 = nn.Linear(64,64)
        self.a3 = nn.Linear(64,64)

        # TransBiGRU
        self.transBiGRU1 = TransBiGRU()
        self.transBiGRU2 = TransBiGRU()
        self.transBiGRU3 = TransBiGRU()

        # classifier
        self.classifier = nn.Linear(64,5)

    def forward(self,graph1,graph2,drug1,drug2,smiles1,smiles2):
        graph = graph1 + graph2
        graph = self.graph_ln(self.graph(graph))
        drug = drug1 + drug2
        drug = self.drug_ln(self.drug(drug))
        smiles = smiles1 + smiles2
        smiles = self.smiles_ln(self.smiles(smiles))

        a1 = F.softmax(self.a1(F.tanh(graph)),dim=-1)
        a2 = F.softmax(self.a2(F.tanh(drug)),dim=-1)
        a3 = F.softmax(self.a3(F.tanh(smiles)),dim=-1)

        multi_feature = a1*graph + a2*drug + a3*smiles
        x = self.transBiGRU3(self.transBiGRU2(self.transBiGRU1(multi_feature)))
        x = self.classifier(x)
        return x