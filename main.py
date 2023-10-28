from preprocess import *
from torch.utils.data import DataLoader
from models import *
import torch.nn as nn
import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
import numpy as np
import random
import torch
from tqdm import tqdm
import os
from transformers import AutoImageProcessor
import torchvision.transforms as T
import time
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint_sequential
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
TOKENIZER = None
def visualize_image(pred, image):
    '''
    Visualizes the original image, then a reconstruction of the the image. Pred and Label are both PyTorch tensors.
    '''
    transform = T.ToPILImage()
    # print(image.shape)
    # print(pred.shape)

    pred = torch.reshape(pred, (16, 3, 32, 32))
    pred_img = transform(pred[0])
    label_img = transform(image[0])

    Image.fromarray(np.hstack((np.array(pred_img),np.array(label_img)))).show()


def visualize_image_epochs(outputs):

    # Initializing subplot counter
    counter = 1

    # Plotting reconstructions
    epochs_list = range(2)

    # Iterating over specified epochs
    for val in epochs_list:
        
        # Extracting recorded information
        temp = outputs[val]['out'].detach().numpy()
        title_text = f"Epoch = {val}"
        
        # Plotting first five images of the last batch
        for idx in range(5):
            plt.subplot(7, 5, counter)
            plt.title(title_text)
            plt.imshow(temp[idx].reshape(28,28), cmap= 'gray')
            plt.axis('off')
            
            # Incrementing the subplot counter
            counter+=1

    # Plotting original images

    # Iterating over first five
    # images of the last batch
    for idx in range(5):
        
        # Obtaining image from the dictionary
        val = outputs[10]['img']
        
        # Plotting image
        plt.subplot(7,5,counter)
        plt.imshow(val[idx].reshape(28, 28),
                cmap = 'gray')
        plt.title("Original Image")
        plt.axis('off')
        
        # Incrementing subplot counter
        counter+=1

    plt.tight_layout()
    plt.show()


def train(data_list, train_loaders, val_loaders, modalities, tasks, load_model=False):
    '''Trains the models on all tasks we want'''


    """ Train Reconstruction of All Modalities
            During Training use modality specific in-caps,
                SUS encoding on every example
                and modality specific reconstruction_cap
        Train Specific Tasks
            Ditch reconstruction_caps and use task specific caps
            Use same modality based in caps and SUS encoding for featurization
        
    """

    ### Assemble Training

    encoding_input_size = 1024
    latent_size = 256
    vocab_size = 3001
    image_input_size = 3*32*32
    text_input_size = 14
    if load_model:
        model = torch.load("./checkpoint-epoch4")
    else:
        modal_map = {"image":(ImageModalityCap, (image_input_size, encoding_input_size)), "text":(TextModalityCap, (vocab_size, latent_size))}
        task_map = {"next_word":(NextWordPredictionCap, (latent_size, vocab_size)), "im_classification": (CategoricalClassificationCap, (latent_size, 10)), 
                    "im_reconstruction": (ReconstructionCap, (latent_size, image_input_size))}

        print("compiling lengths and ")
        lengths = [int(len(train_loaders[loader])) for loader in data_list]
        print("finished lengths")
        
        random_batching = np.concatenate(([[idx for _ in range(length)] for idx, length in enumerate(lengths)]), dtype=int)
        np.random.shuffle(random_batching)
        print("got batches")
        
        sus_encoding = SUSEncoding(encoding_input_size, latent_size)
        modality_caps = nn.ModuleList([modal_map[modality][0](*modal_map[modality][1]) for modality in modalities])
        task_caps = nn.ModuleList([task_map[task][0](*task_map[task][1]) for task in tasks])
        print("assembled caps")
        model = SUSModel(modality_caps, modalities, task_caps, tasks, sus_encoding)
        print("constructed model")    
        
    
    # train_loader.to(device)
    # val_loader.to(device)
    # OR  GPU STUFF:
    # gpu = ?
    # torch.cuda.set_device(gpu)
    # device = torch.device('cuda:' + str(gpu + args.offset)
    # torch.distributed.init_process_group(backend='nccl', world_size=N, init_method='...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    model = model.to(device)

    model.train()
    outputs = {}

    for epoch in range(5):
        print('epoch: ', epoch)
        i = 0
        allLoss = 0
        reconLoss = 0
        recons = 0
        classif = 0
        task_metrics = {task: {'loss':0, 'acc':0, 'count' :0} for task in tasks}
        iterators = [iter(train_loaders[loader]) for loader in data_list]
        print("finished iters")
        for batch in tqdm(random_batching):
            
            input = next(iterators[batch])
            
            
            
            #segments = 2
            #modules = [module for k, module in model._modules.items()]
            
            if modalities[batch] == "text":
                
                sequence = torch.squeeze(input, 1)

                sequence = torch.reshape(sequence, [sequence.shape[0]*10, 15])
                input = sequence[:, :14]
                labels = sequence[:, 14].long()
                input = sequence.int()
                input = input.to(device)
                labels = labels.to(device)
            elif modalities[batch] == "image":
                input, labels = input[0].to(device), input[1].to(device)
            
            #pred = checkpoint_sequential(modules, segments, (input, modalities[batch], tasks[batch]))
            
            pred = model(input, modalities[batch], tasks[batch])
            if tasks[batch]  == "im_reconstruction":
                loss = model.loss_fn(pred, input)
                task_metrics[tasks[batch]]['loss'] += loss
                task_metrics[tasks[batch]]['count'] += 1
            else:
                loss = model.loss_fn(pred, labels)
                task_metrics[tasks[batch]]['loss'] += loss
                task_metrics[tasks[batch]]['count'] += 1
                task_metrics[tasks[batch]]['acc'] += model.acc_fn(pred, labels)
                

            if i != 0 and i % 100 == 0:
                for task in tasks:
                    print(f"Task {task}: Loss {task_metrics[task]['loss']/task_metrics[task]['count']}, Acc {task_metrics[task]['acc']/task_metrics[task]['count']}")
                    task_metrics[task]["loss"] = 0
                    task_metrics[task]["count"] = 0
                    task_metrics[task]["acc"] = 0
                #visualize_image(pred, input)
                allLoss = 0
                reconLoss = 0
                classif = 0 
                recons = 0


            i +=1
            

            # loss stuff
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step() 
        
        # Storing useful images and
        # reconstructed outputs for the last batch
        torch.save(model, f"./checkpoint-epoch{epoch}")
    return model

def test_model(val_loaders, data_list, modalities, tasks, model=None):
    # Test the model

    #val_model = torch.load('./textgen_checkpoint')
    val_model = model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    i_val = 0
    with torch.no_grad():
        for idx, data in enumerate(data_list):
            loader = val_loaders[data]
            modality = modalities[idx]
            task = tasks[idx]
            val_allAcc = 0
            val_allLoss = 0
            for input in loader:
                
                if modality == "text":
                
                    sequence = torch.squeeze(input, 1)

                    sequence = torch.reshape(sequence, [sequence.shape[0]*10, 15])
                    input = sequence[:, :14]
                    labels = sequence[:, 14].long()
                    input = sequence.int()
                    input = input.to(device)
                    labels = labels.to(device)
                elif modality == "image":
                    input, labels = input[0].to(device), input[1].to(device)

                # inputs, labels = inputs.to(device), labels.to(device)
                pred = val_model(input, modality, task)
                if task  == "im_reconstruction":
                    val_loss = val_model.loss_fn(pred, input)
                    val_acc = 0
                    val_allAcc += val_acc
                    val_allLoss += val_loss
                    if i_val != 0 and i_val % 100 == 0:
                        visualize_image(pred, input)
                else:
                    #val_loss = val_model.loss_fn(pred, labels)
                    val_acc = val_model.acc_fn(pred, labels)
                    val_allAcc += val_acc
                    #val_allLoss += val_loss
                if i_val != 0 and i_val % 100 == 0:
                    #print(val_loss)
                    print(val_acc)
                i_val += 1
            print(f"Task {task} Loss {0} Acc {val_allAcc/len(loader)}")

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_loaders_off_disk(list_of_data):
    global TOKENIZER
    ''' [WikiText, cifar, mscoco]'''
    data_dict = {"WikiText":get_WikiText_dataloaders, "cifar":get_cifar_dataloaders, "mscoco":get_mscoco_dataloaders}
    train_loaders = {}
    test_loaders = {}
    for i in set(list(list_of_data)):
        if i =="WikiText":
            trainset, testset, TOKENIZER = data_dict[i]()
        else:
            trainset, testset = data_dict[i]()
        train_loaders[i] = trainset
        test_loaders[i] = testset
    return train_loaders, test_loaders

def get_loaders_on_disk(list_of_data):
    global TOKENIZER
    data_dict = {"WikiText":"./data/WikiText_", "cifar":"./data/cifar_", "mscoco":"./data/msoco_"}
    train_loaders = {}
    test_loaders = {}
    for i in set(list(list_of_data)):
        if i == "WikiText":
            TOKENIZER = torch.load(data_dict[i]+"tokenizer")
        train_loader = torch.load(data_dict[i]+"train") 
        test_loader = torch.load(data_dict[i]+"test")
        train_loaders[i] = train_loader
        test_loaders[i] = test_loader
    return train_loaders, test_loaders

SAVE_DATA = False
TRAIN_FOLDER = "./data"
TEST_FOLDER = "./data"
def main():
    
    data_list = ["cifar", "cifar"]
    modalities = ["image", "image"]
    tasks = ["im_classification", "im_reconstruction"]
    if SAVE_DATA:
        
        train_loaders, test_loaders = get_loaders_off_disk(data_list)
        for data in data_list:
            if TOKENIZER != None and data == "WikiText":
                torch.save(train_loaders[data], TRAIN_FOLDER+"/"+data+"_tokenizer")
            torch.save(train_loaders[data], TRAIN_FOLDER+"/"+data+"_train")
            torch.save(test_loaders[data], TEST_FOLDER+"/"+data+"_test")
    else:
        train_loaders, test_loaders = get_loaders_on_disk(data_list)
        
    model = train(data_list, train_loaders, test_loaders, modalities, tasks)
    test_model(test_loaders, data_list, modalities, tasks, model)
    
if __name__ == "__main__":
    set_seed()
    main()