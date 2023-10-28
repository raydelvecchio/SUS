import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from os.path import exists
import pickle
import torch
import numpy as np
import re
from collections import Counter
from transformers import BertTokenizer


TRAIN_DATAPATH="./data/train2017"
TRAIN_JSONPATH="./data/annotations/captions_train2017.json"

VAL_DATAPATH="./data/val2017/val2017"
VAL_JSONPATH="./data/annotations/captions_val2017.json"

BATCH_SIZE = 16
NUM_WORKERS = 4

VAL_PICKLE_PATH = 'val_loader.pkl'
TRAIN_PICKLE_PATH = 'train_loader.pkl'

BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased", vocab_size = 3000) 


class Custom_Tokenizer():
    def __init__(self, max_vocab=2999, unk_idx = 3000) -> None:
        self.max_vocab = max_vocab
        self.unk_idx = unk_idx

        self.word_to_idx = {}
        self.idx_to_word = {}

        self.word_to_idx['UNK'] = unk_idx
        self.idx_to_word[unk_idx] = 'UNK'

        self.regex = r'\w+|[^\w\s]+'  # regex to tok

    def __get_tokens(self, text: str):
        return re.findall(self.regex, text)

    def __convert_to_idx(self, token):
        if token in self.word_to_idx:
            return self.word_to_idx[token]
        else:
            return self.unk_idx
    
    def fit(self, all_text_corpus: str):
        """
        Populates the word_to_idx and idx_to_word on the training corpus of words.
        """
        tokens = self.__get_tokens(all_text_corpus)  # splits on whitespace and all punctuation
        counter = Counter(tokens)
        most_common = counter.most_common(self.max_vocab)
        most_common_tokens = [x[0] for x in most_common]
        for i, tok in enumerate(most_common_tokens):
            self.word_to_idx[tok] = int(i)
            self.idx_to_word[int(i)] = tok

    def encode(self, text: str, max_length=150, padding_token=1):
        """
        Encodes the words and returns a PyTorch tensor of the word indices.
        """    
        tokens = self.__get_tokens(text)
        tokens = [self.__convert_to_idx(token) for token in tokens]
        if len(tokens) >= max_length:
            tokens = tokens[:max_length]
        else:
            while(len(tokens)) < max_length:
                tokens.append(padding_token)
        return torch.Tensor(tokens)

    def decode(self, indices: list):
        return [self.idx_to_word[idx] for idx in indices]


# custom class to ensure each image has same number of labels associated with it
class PaddedCocoDetection(dset.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, max_labels=10, max_tokens=20, pad_value="shit", pad_num=1):
        super().__init__(root, annFile, transform, target_transform)
        self.max_labels = max_labels
        self.max_tokens = max_tokens
        self.pad_value = pad_value
        self.pad_num = pad_num

    def __tokenize(self, words: str):
        return BERT_TOKENIZER.convert_tokens_to_ids(BERT_TOKENIZER.tokenize(words))
    
    def __pad_caption(self, caption_ids: list):
        return caption_ids[:self.max_tokens] + [self.pad_num] * (self.max_tokens - len(caption_ids))

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # Flatten the labels and extract the category ids
        labels = [obj['caption'] for obj in target]

        # Pad the labels with the pad_value
        padded_labels = labels[:self.max_labels] + [self.pad_value] * (self.max_labels - len(labels))
        padded_labels = [self.__tokenize(label) for label in padded_labels]
        padded_labels = [self.__pad_caption(caption) for caption in padded_labels]

        # Convert the padded labels to a tensor
        padded_labels_tensor = torch.tensor(padded_labels, dtype=torch.long)

        return img, padded_labels_tensor


def load_mscoco_dataset() -> tuple[Dataset, Dataset]:
    '''returns the msCOCO dataset objects for training and testing'''
    print("Loading Data")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # images are of size 3x640x480
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    coco_train = PaddedCocoDetection(root = TRAIN_DATAPATH, annFile = TRAIN_JSONPATH, transform=transform)
    coco_val = PaddedCocoDetection(root = VAL_DATAPATH, annFile = VAL_JSONPATH)
    return coco_train, coco_val

def get_mscoco_dataloaders() -> tuple[DataLoader, DataLoader]:
    '''given two datasets for train and test, creates the dataloaders. Pickles them as a file if they haven't been pickled, or loads 
    from pickles if they are.''' 
    if exists(VAL_PICKLE_PATH) and exists(TRAIN_PICKLE_PATH):
        with open(VAL_PICKLE_PATH, "rb") as f:
            val_loader = pickle.load(f)
        with open(TRAIN_PICKLE_PATH, "rb") as f:
            train_loader = pickle.load(f)
    else:
        train, val = load_mscoco_dataset()
        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
        val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
        with open(VAL_PICKLE_PATH, "wb") as f:
            pickle.dump(val_loader, f)
        with open(TRAIN_PICKLE_PATH, "wb") as f:
            pickle.dump(train_loader, f)
    return train_loader, val_loader

def get_cifar_dataloaders():
    print("Loading CIFAR...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = CIFAR10(root='./data/cifar', train=True,
                                        download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    random_inds = np.random.choice(len(trainset), size=20000, replace=False)
    trainset = torch.utils.data.Subset(trainset, random_inds)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    return train_loader, val_loader

def get_WikiText_dataloaders():
    """
    Returns dataloaders for WikiText2 dataset, as well as the tokenizer so we can decode.
    """
    print("Loading WikiText2...")
    trainset, testset = WikiText2(split=('train', 'test'))

    # collecting the string corpus of all data
    all_text = ""
    for w in trainset:
        all_text += w
    for w in testset: 
        all_text += w

    trainset, testset = WikiText2(split=('train', 'test'))  # loading in again since we ran out of the generator in the loops above

    tokenizer = Custom_Tokenizer()
    tokenizer.fit(all_text_corpus=all_text)

    trainset = [tokenizer.encode(t) for t in trainset]
    testset = [tokenizer.encode(t) for t in testset]

    random_inds = np.random.choice(len(trainset), size=10000, replace=False)
    trainset = torch.utils.data.Subset(trainset, random_inds)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    return train_loader, val_loader, tokenizer
