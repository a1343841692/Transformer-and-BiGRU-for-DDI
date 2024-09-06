import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import pandas
import torch
import torch.nn as nn
from Model import DDI_MODEL
import torch.optim as optim

# PCA
from sklearn.decomposition import PCA
PCA5 = PCA(n_components=5)
PCA12 = PCA(n_components=12)
PCA256 = PCA(n_components=256)

# graph2vec
import torch_geometric.nn as pygnn
import torch_geometric.utils.smiles as S

# word2vec
import gensim.downloader
from gensim.models import KeyedVectors
glove_vectors = gensim.downloader.load('glove-twitter-25')

# smiles2vec
from transformers import AutoTokenizer, AutoModelForPreTraining
tokenizer = AutoTokenizer.from_pretrained("unikei/bert-base-smiles")
model = AutoModelForPreTraining.from_pretrained("unikei/bert-base-smiles")


class DDI_Dataset(Dataset):
    def __init__(self,root):

        # ['Unnamed: 0', 'text', 'drug1', 'drug2', 'ddi', 'type', 'label', 'smile1', 'smile2', 'pos1', 'pos2']
        csv_file = pd.read_csv(path)

        # data info list
        self.drug1_list = csv_file['drug1'].tolist()
        self.drug2_list = csv_file['drug2'].tolist()
        self.smile1_list = csv_file['smile1'].tolist()
        self.smile2_list = csv_file['smile2'].tolist()
        self.label_list = csv_file['label'].tolist()


    def __getitem__(self, index):

        # drug2vec
        try:
            drug1 = glove_vectors[self.drug1_list[index]]
        except:
            drug1 = np.zeros(25)
        try:
            drug2 = glove_vectors[self.drug2_list[index]]
        except:
            drug2 = np.zeros(25)

        # smiles2vec
        try:
            smiles1 = self.smile1_list[index]
            tokens = tokenizer(smiles1, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
            smiles1 = model(**tokens)[0][0,-1,:].detach().numpy()
        except:
            smiles1 = np.zeros(30000)

        try:
            smiles2 = self.smile2_list[index]
            tokens = tokenizer(smiles2, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
            smiles2 = model(**tokens)[0][0, -1, :].detach().numpy()
        except:
            smiles2 = np.zeros(30000)

        # graph2vec
        try:
            data1 = S.from_smiles(self.smile1_list[index])
            batch1 = torch.zeros(len(data1.x)).long()
            graph1 = pygnn.global_mean_pool(data1.x,batch1).detach().numpy().reshape(-1)
        except:
            graph1 = np.zeros(9)

        try:
            data2 = S.from_smiles(self.smile2_list[index])
            batch2 = torch.zeros(len(data2.x)).long()
            graph2 = pygnn.global_mean_pool(data2.x, batch2).detach().numpy().reshape(-1)
        except:
            graph2 = np.zeros(9)



        return (
            graph1,graph2,
            drug1,drug2,
            smiles1,smiles2,
            self.label_list[index]
                )

    def __len__(self):
        return len(self.label_list)

if __name__ == '__main__':


    path = r'C:\Users\yc096\PycharmProjects\pythonProject\Dataset\train_smiles_pos.csv' # your path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DDI_MODEL().to(device)
    dataset = DDI_Dataset(root=path)
    batch_size = 8
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练开始
    for epoch in range(1, 200 + 1):

        # 一次训练
        model.train()
        train_loss = 0
        iter = 0
        for datas in dataloader:
            optimizer.zero_grad()
            graph1,graph2,drug1,drug2,smiles1,smiles2,labels = datas
            graph1 = graph1.float().to(device)
            graph2 = graph2.float().to(device)
            drug1 = drug1.float().to(device)
            drug2 = drug2.float().to(device)
            smiles1 = smiles1.float().to(device)
            smiles2 = smiles2.float().to(device)
            labels = labels.long().to(device)
            preds = model(graph1,graph2,drug1,drug2,smiles1,smiles2)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iter+=1
            # continue    # debug point
        print( 'EPOCH:{} train_loss:{:.4f}'.format(epoch,train_loss/iter))