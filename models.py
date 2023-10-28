import torch.nn as nn
import torch
import numpy as np
import time 
from transformers import SwinModel, BertModel


class TaskSpecificCap(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # input size should be the OUTPUT size of the SUS embedding
        self.cap = nn.Linear(input_size, output_size, bias=True)
    
    def forward(self, x):
        return self.cap(x)


class ImageModalityCap(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # output size should be the INPUT size of the SUS embedding
        #self.cap = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.conv1 = nn.Conv2d(3, 12, 4)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(12, 24, 4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2904, output_size)
        self.cap = nn.Sequential(self.conv1, self.pool, self.conv2)
        self.out = nn.Sequential(self.flatten, self.fc)
    def forward(self, x: torch.tensor):
        x = self.cap(x)
        out = self.out(x)
        return out


class TextModalityCap(nn.Module):
    def __init__(self, vocab_size, latent_encoding_size, hidden_size=360):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_encoding_size = latent_encoding_size
    
        #[batch,seq, embed]
        embedding_layer = nn.Embedding(self.vocab_size, 24)
        encoder_layer = nn.TransformerEncoderLayer(d_model=24, nhead=4)
        encoder = nn.TransformerEncoder(encoder_layer, 1)
        self.flatten = nn.Flatten()
        self.cap = nn.Sequential(embedding_layer, encoder, 
                                 self.flatten, 
                                   nn.Linear(self.hidden_size, latent_encoding_size))

    def forward(self, input_ids):
        out = self.cap(input_ids)
        return out


class ReconstructionCap(nn.Module):
    def __init__(self, latent_size, input_size, hidden_size = 1024):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decoder = nn.Sequential(nn.Linear(latent_size, self.hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(self.hidden_size, self.hidden_size*3),
                            nn.LeakyReLU(),
                            nn.Linear(self.hidden_size*3, self.input_size),
                            nn.Sigmoid())

        self.mse_loss = nn.MSELoss(reduction='sum')
    
    def forward(self, x: torch.tensor):
        return self.decoder(x)
    
    def loss(self, recon_x, x):
        x = torch.flatten(x, start_dim=1)
        mse_loss = self.mse_loss(recon_x, x)
        
        return mse_loss/len(x)

    
class CategoricalClassificationCap(nn.Module):
    def __init__(self, latent_size, number_classes, hidden_size = None):
        super().__init__()
        self.latent_size = latent_size
        self.number_classes = number_classes
        if hidden_size == None:
            self.hidden_size = latent_size
        else:
            self.hidden_size = hidden_size
        
        self.classification_head = nn.Sequential(nn.Linear(self.latent_size, self.number_classes)
                                                   )
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, x):
        out = self.classification_head(x)
        return out

    def loss(self, pred, true):
        return self.cross_entropy(pred, true)

    def acc(self, pred, true):
        total = 0
        correct = 0
        predicted_class = torch.argmax(pred, dim=1)
        total += true.size(0)
        correct += (predicted_class == true).sum().item()
        return correct / total

class NextWordPredictionCap(nn.Module):
    def __init__(self, latent_size, vocab_size, hidden_size = 1024):
        super().__init__()
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        if hidden_size == None:
            self.hidden_size = latent_size*2
        else:
            self.hidden_size = hidden_size
        self.classification_head = nn.Sequential(nn.Linear(self.latent_size, self.hidden_size),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(self.hidden_size, self.vocab_size),
                                                    )
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, x):
        out = self.classification_head(x)
        return out

    def loss(self, pred, true):
        return self.cross_entropy(pred, true)

    def acc(self, pred, true):
        # compute perplexity
        return torch.exp(torch.mean(self.loss(pred, true)))
    
class SUSEncoding(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size=512):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.sequential = nn.Sequential(nn.Linear(input_size, self.hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(self.hidden_size, self.hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(self.hidden_size, self.latent_size),
                            nn.LeakyReLU())  # shrinking the size as we go down
    def forward(self, x: torch.tensor):
        x = self.sequential(x)
        return x


class SUSModel(nn.Module):
    def __init__(self, modality_caps, modalities, task_caps, tasks, sus_embedding):
        '''
        Params: modalityCaps: A list of tuple pairings, where the first item is
                                the cap and the second is its corresponding modality str.
                                modality strings: ["image", "text", "audio"]
                taskCaps: A list of tuple pairings, where the first item is 
                                the cap and the second is its corresponding task str.
                                task strings: ["class", "caption", "reconstruction"]
                batch_size: batch_size for training. 
        '''
        super().__init__()
        self.modality_caps = modality_caps
        self.modality_list = modalities
        self.task_caps = task_caps
        self.task_list = tasks
        self.sus_embedding = sus_embedding
        self.modal_cap = None
        self.task_cap = None

    def get_caps(self, inp_modal: str, task: str):
        '''This gathers the caps needed for a forward pass'''
        # print(self.task_list)
        # print(self.modality_caps, self.modality_list)
        # print("task: ", task)
        task_index = self.task_list.index(task)
        task_cap = self.task_caps[task_index]
        modal_index = self.modality_list.index(inp_modal)
        modal_cap = self.modality_caps[modal_index]
        return modal_cap, task_cap
    
    def forward(self, x: torch.tensor, inp_modal: str, task:str):
        self.modal_cap, self.task_cap = self.get_caps(inp_modal, task)

        in_cap = self.modal_cap(x)
        
        sus = self.sus_embedding(in_cap)

        out_cap = self.task_cap(sus)
        return out_cap
    
    def loss_fn(self, output: torch.tensor, label):
        
        loss = self.task_cap.loss(output, label)
        return loss

    def acc_fn(self, output: torch.tensor, label):
        acc = self.task_cap.acc(output, label)
        return acc
        
