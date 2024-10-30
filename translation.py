from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchtext
import torch
from torch.utils.data import random_split
from collections import defaultdict

class Translation(Dataset):
    def __init__(self, source_file, target_file, train_size=0.9):
        self.ingles = []
        self.espanol = []
        self.tokenizer_es = get_tokenizer("spacy", language="es_core_news_md")
        self.tokenizer_en = get_tokenizer("spacy", language="en_core_web_md")
        self.vocab_es = torchtext.vocab.FastText(language='es', unk_init=torch.Tensor.normal_)
        self.vocab_en = torchtext.vocab.FastText(language='en', unk_init=torch.Tensor.normal_)

        self.vocab_en = self.add_sos_eos_unk_pad(self.vocab_en)
        self.vocab_es = self.add_sos_eos_unk_pad(self.vocab_es)

        self.archivo_ingles = source_file
        self.archivo_espanol = target_file

        # Leer el conjunto de datos
        for ingles, espanol in self.read_translation():
            self.ingles.append(ingles)
            self.espanol.append(espanol)
        
        # Dividir en entrenamiento y test
        train_size = int(len(self) * train_size)
        test_size = len(self) - train_size
        self.train_dataset, self.test_dataset = random_split(self, [train_size, test_size])



    def add_sos_eos_unk_pad(self, vocabulary):
        words = vocabulary.itos
        vocab = vocabulary.stoi
        embedding_matrix = vocabulary.vectors

        # Tokens especiales
        sos_token = '<sos>'
        eos_token = '<eos>'
        pad_token = '<pad>'
        unk_token = '<unk>'

        # Inicializamos los vectores para los tokens especiales, por ejemplo, con ceros
        sos_vector = torch.full((1, embedding_matrix.shape[1]), 1.)
        eos_vector = torch.full((1, embedding_matrix.shape[1]), 2.)
        pad_vector = torch.zeros((1, embedding_matrix.shape[1]))
        unk_vector = torch.full((1, embedding_matrix.shape[1]), 3.)

        # Añade los vectores al final de la matriz de embeddings
        embedding_matrix = torch.cat((embedding_matrix, sos_vector, eos_vector, unk_vector, pad_vector), 0)

        # Añade los tokens especiales al vocabulario
        vocab[sos_token] = len(vocab)
        vocab[eos_token] = len(vocab)
        vocab[pad_token] = len(vocab)
        vocab[unk_token] = len(vocab)

        words.append(sos_token)
        words.append(eos_token)
        words.append(pad_token)
        words.append(unk_token)

        vocabulary.itos = words
        vocabulary.stoi = vocab
        vocabulary.vectors = embedding_matrix

        default_stoi = defaultdict(lambda : len(vocabulary)-1, vocabulary.stoi)
        vocabulary.stoi = default_stoi
    
        return vocabulary
        

    def read_translation(self):
        with open(self.archivo_ingles, 'r', encoding='utf-8') as f_ingles, open(self.archivo_espanol, 'r', encoding='utf-8') as f_espanol:
            for oracion_ingles, oracion_espanol in zip(f_ingles, f_espanol):
                yield oracion_ingles.strip().lower(), oracion_espanol.strip().lower()

    def __len__(self):
        return len(self.ingles)

    def __getitem__(self, idx):
        item = self.ingles[idx], self.espanol[idx]
        tokens_ingles = self.tokenizer_en(item[0])
        tokens_espanol = self.tokenizer_es(item[1])

        tokens_ingles = tokens_ingles + ['<eos>']
        tokens_espanol = ['<sos>'] + tokens_espanol + ['<eos>']

        if not tokens_ingles or not tokens_espanol:
            return torch.zeros(1, 300), torch.zeros(1, 300)
            # raise RuntimeError("Una de las muestras está vacía.")
    
        tensor_ingles = self.vocab_en.get_vecs_by_tokens(tokens_ingles)
        tensor_espanol = self.vocab_es.get_vecs_by_tokens(tokens_espanol)

        indices_ingles = [self.vocab_en.stoi[token] for token in tokens_ingles] + [self.vocab_en.stoi['<pad>']]
        indices_espanol = [self.vocab_es.stoi[token] for token in tokens_espanol] + [self.vocab_es.stoi['<pad>']]

        return tensor_ingles, tensor_espanol, indices_ingles, indices_espanol
        
            
        
def collate_fn(batch):
    ingles_batch, espanol_batch, ingles_seqs, espanol_seqs = zip(*batch)
    ingles_batch = pad_sequence(ingles_batch, batch_first=True, padding_value=0)
    espanol_batch = pad_sequence(espanol_batch, batch_first=True, padding_value=0)

    # Calcular la longitud máxima de la lista de listas de índices
    pad = espanol_seqs[0][-1]  # token <pad>
    max_len = max([len(l) for l in espanol_seqs])
    for seq in espanol_seqs:
        seq += [pad]*(max_len-len(seq))
        
    return ingles_batch, espanol_batch, ingles_seqs, espanol_seqs