###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/positive',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./models/model_pos.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--input', type=str, default = "",
                    help='input data')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
starter = args.input #get start sequence

#add check if in corpus
starter_ids = []
if len(starter) >0: #check if an input sequence was given
    for i in starter.split(): #check which words are in the corpus
        try:
            starter_ids.append(corpus.dictionary.word2idx[i.lower()])
        except: #if not in corpus skip them and give feedback!
            print("The following word is not in the corpus and will be skipped: " + i.lower())
    
    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
    if not is_transformer_model:
        hidden = model.init_hidden(max([len(starter_ids), 1])) #if no word in corpus, choose one as length
    if len(starter_ids)>0: 
        input = torch.tensor([starter_ids]).type(torch.int64)
    else: #if no input word in sequence, give feed back and proceed as if no sequence was given
        print("No word is in the corpus...\nProceeding with random initiation!")
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

else: #no input sequence, initialise randomly
    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
    if not is_transformer_model:
        hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
    # print input sequence:
        for i, idx in enumerate(starter_ids):
            outf.write(corpus.dictionary.idx2word[idx] + ('\n' if i % 20 == 19 else ' '))
        for i in range(len(starter_ids), args.words):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.flatten(torch.multinomial(word_weights, 1))[0] #choose first element
                input.fill_(word_idx)
            
            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' ')) #line break every 20 words

            if (i+1) % args.log_interval == 0: #give message every n words
                print('| Generated {}/{} words'.format(i+1, args.words))
                
                
# not completely sure about the nature of input in the model() function
# I initialise input with the indices of the starting sequence
# Afterwards for each new word i only take the first value given from the torch.multinomial function (flattened)
# I guess it kinda works...
# The initial hidden layer depends on the length of the input sequence