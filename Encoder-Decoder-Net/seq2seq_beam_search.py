#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import time
import pynvml
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
import operator
from queue import PriorityQueue
from log import timeit
from pytorchtools import EarlyStopping

SOS_token = 2
EOS_token = 3
# In[ ]:


def base_structure():
    # data read
    x_train=np.load(r'./data_split/x_train.npy')
    x_test=np.load(r'./data_split/x_test.npy')
    x_validation=np.load(r'./data_split/x_validation.npy')
    y_train=np.load(r'./data_split/y_train.npy')
    y_test=np.load(r'./data_split/y_test.npy')
    y_validation=np.load(r'./data_split/y_validation.npy')
    print(x_train)
    print(y_train)    
    #data standard normalization
    a,b,c=x_train.shape
    x_train=x_train.reshape(a*b,c)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    x_train=x_train.reshape(a,b,c)

    a,b,c=x_validation.shape
    x_validation=x_validation.reshape(a*b,c)
    x_validation=scaler.transform(x_validation)
    x_validation=x_validation.reshape(a,b,c)

    a,b,c=x_test.shape
    x_test=x_test.reshape(a*b,c)
    x_test=scaler.transform(x_test)
    x_test=x_test.reshape(a,b,c)

    x1=torch.from_numpy(x_train).float()
    y1=torch.from_numpy(y_train).float()
    x2=torch.from_numpy(x_validation).float()
    y2=torch.from_numpy(y_validation).float()
    x3=torch.from_numpy(x_test).float()
    y3=torch.from_numpy(y_test).float()
    
    #data from.npy to pytorch data

    global BATCH_SIZE
    BATCH_SIZE=512


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = Data.TensorDataset(x1,y1)

    trainloader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False, # 要不要打乱数据 (打乱比较好)
	num_workers=2,              # 多线程来读数据
        drop_last=True,
    )



    vali_dataset = Data.TensorDataset(x2,y2)

    valiloader = Data.DataLoader(
        dataset=vali_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False, # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        drop_last=True,
    )


    test_dataset = Data.TensorDataset(x3,y3)

    testloader = Data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False, # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
        drop_last=True,
    )
    
#encoder str

    class Encoder(nn.Module):
        def __init__(self, input_dim,emb_dim, hid_dim, n_layers, dropout=0.1):
            super().__init__()

            self.input_dim = input_dim
            self.emb_dim = emb_dim
            self.hid_dim = hid_dim
            self.n_layers = n_layers
            self.dropout = dropout

            self.embedding = nn.Linear(input_dim, emb_dim)

            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

            self.dropout = nn.Dropout(dropout)

        def forward(self, x):

            #x = [len, batch size,inout_size]


            embedded = self.dropout(self.embedding(x))

            #embedded = [len, batch size, emb dim]

            outputs, (hidden, cell) = self.rnn(embedded)

            #outputs = [src sent len, batch size, hid dim * n directions]
            #hidden = [n layers * n directions, batch size, hid dim]
            #cell = [n layers * n directions, batch size, hid dim]

            #outputs are always from the top hidden layer

            return hidden, cell
# decoder str

    class Decoder(nn.Module):
        def __init__(self, decoder_input_dim,emb_dim, hid_dim, n_layers, dropout=0.2):
            super().__init__()

            self.emb_dim = emb_dim
            self.hid_dim = hid_dim
            self.decoder_input_dim = decoder_input_dim
            self.n_layers = n_layers
            self.dropout_rate = dropout

            self.embedding = nn.Linear(decoder_input_dim, emb_dim)
            #self.embedding = nn.Embedding(decoder_input_dim, emb_dim)
            self.rnn = nn.LSTM(emb_dim+hid_dim, hid_dim, n_layers, dropout = dropout)

            self.out = nn.Linear(hid_dim, decoder_input_dim)

            self.dropout = nn.Dropout(dropout)

        def forward(self, input, context, hidden, cell):

            #embedded = self.embedding(input.long())
            #embedded = self.dropout(embedded.transpose(0, 1))
            #embedded = embedded.transpose(0, 1) 
            input = input.unsqueeze(0)
            #embedded = F.dropout(embedded, self.dropout_rate)
            #input = input.unsqueeze(0)
            embedded = torch.squeeze(self.dropout(self.embedding(input))).unsqueeze(0)
            #input = [1, batch size]
    #         print('inputshape:',input.shape)
            emb_con = torch.cat((embedded, context), dim = 2)
            ##embedded = self.dropout(self.embedding(input))

            #embedded = [1, batch size, emb dim]
            #emb_con = torch.cat((embedded, context), dim = 2)
            output, (hidden, cell) = self.rnn(emb_con, (hidden, cell))

            #output = [len, batch size, hid dim * n directions]
            #hidden = [n layers * n directions, batch size, hid dim]
            #cell = [n layers * n directions, batch size, hid dim]

            #sent len and n directions will always be 1 in the decoder, therefore:
            #output = [1, batch size, hid dim]
            #hidden = [n layers, batch size, hid dim]
            #cell = [n layers, batch size, hid dim]
            prediction = self.out(output)

            #prediction = [batch size, output dim]

            return prediction, hidden, cell

        
    class Seq2Seq(nn.Module):
        global firstinput
        def __init__(self, encoder, decoder, device):
            super().__init__()

            self.encoder = encoder

            self.decoder = decoder
            self.device = device

            assert encoder.hid_dim == decoder.hid_dim,             "Hidden dimensions of encoder and decoder must be equal!"
            assert encoder.n_layers == decoder.n_layers,             "Encoder and decoder must have equal number of layers!"

        def forward(self, x, y, teacher_forcing_ratio = 0.5):

            #src = [src sent len, batch size]
            #trg = [trg sent len, batch size]
            #teacher_forcing_ratio is probability to use teacher forcing
            #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

            batch_size = BATCH_SIZE
            max_len = 25
            trg_vocab_size = 2

            #tensor to store decoder outputs
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size)

            #last hidden state of the encoder is used as the initial hidden state of the decoder
            hidden, cell = self.encoder(x)
            context=cell[1,:,:]
            context=context.unsqueeze(0)
    #         print('c-shape:',context.shape)
            #first input to the decoder is the <sos> tokens
            input=firstinput
            #print(input.size())
    #         input = input.unsqueeze(0)
            #print(input.size())
            for t in range(max_len):

                output, hidden, cell = self.decoder(input, context,hidden, cell)
                outputs[t] = output
                #print(output)
                #input = output.unsqueeze(0)
                #print(input.size())
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output
                if t==24:
                    break
                input = ((y[t,:,:]) if teacher_force else top1)
                #outputs[t] = output
                #print('output',output.size())
                #input = output.unsqueeze(0)


            return outputs
        
        def decode(self, src, trg, method='beam-search'):
            encoder_output, hidden = self.encoder(src)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
            hidden = hidden[:self.decoder.n_layers]  # [4, 32, 512][1, 32, 512]
            if method == 'beam-search':
                self.beam_decode(trg, hidden, encoder_output)
            else:
                self.greedy_decode(trg, hidden, encoder_output)
        
        def greedy_decode(self, trg, decoder_hidden, encoder_outputs, ):
       	    '''
            :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
            :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
            :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
            :return: decoded_batch
            '''
            seq_len, batch_size = trg.size()
            decoded_batch = torch.zeros((batch_size, seq_len))
            # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).cuda()
            decoder_input = Variable(trg.data[0, :]).cuda()  # sos
            print(decoder_input.shape)
            for t in range(seq_len):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

                topv, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
                topi = topi.view(-1)
                decoded_batch[:, t] = topi

                decoder_input = topi.detach().view(-1)

            return decoded_batch

        @timeit
        def beam_decode(self, target_tensor, decoder_hiddens, encoder_outputs=None):
            '''
            :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
            :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
            :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
            :return: decoded_batch
            '''
            target_tensor = target_tensor.permute(1, 0)
            beam_width = 10
            topk = 1

  # how many sentence do you want to generate
            decoded_batch = []

            # decoding goes sentence by sentence
            for idx in range(target_tensor.size(0)):  # batch_size
                if isinstance(decoder_hiddens, tuple):  # LSTM case
                    decoder_hidden = (
                        decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
                else:
                    decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)  # [1, B, H]=>[1,H]=>[1,1,H]
                    encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)  # [T,B,H]=>[T,H]=>[T,1,H]

                # Start with the start of the sentence token
                decoder_input = torch.LongTensor([SOS_token]).cuda()

                # Number of sentence to generate
                endnodes = []
                number_required = min((topk + 1), topk - len(endnodes))

                # starting node -  hidden vector, previous node, word id, logp, length
                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
                nodes = PriorityQueue()

                # start the queue
                nodes.put((-node.eval(), node))
                qsize = 1

                # start beam search
                while True:
                    # give up when decoding takes too long
                    if qsize > 2000: break

                    # fetch the best node
                    score, n = nodes.get()
                    # print('--best node seqs len {} '.format(n.leng))
                    decoder_input = n.wordid
                    decoder_hidden = n.h

                    if n.wordid.item() == EOS_token and n.prevNode != None:
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    # decode for one step using decoder
                    decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)

                    # PUT HERE REAL BEAM SEARCH OF TOP
                    log_prob, indexes = torch.topk(decoder_output, beam_width)
                    nextnodes = []
                    for new_k in range(beam_width):
                        decoded_t = indexes[0][new_k].view(-1)
                        log_p = log_prob[0][new_k].item()

                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval()
                        nextnodes.append((score, node))

                    # put them into queue
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))
                        # increase qsize
                    qsize += len(nextnodes) - 1

                # choose nbest paths, back trace them
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for _ in range(topk)]

                utterances = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.wordid)
                    # back trace
                    while n.prevNode != None:
                        n = n.prevNode
                        utterance.append(n.wordid)

                    utterance = utterance[::-1]
                    utterances.append(utterance)

                decoded_batch.append(utterances)

            return decoded_batch

    class BeamSearchNode(object):
        def __init__(self, hiddenstate, previousNode, wordId, logProb, length):

            '''
	    :param hiddenstate:
	    :param previousNode:
	    :param wordId:
	    :param logProb:
	    :param length:
	    '''
            self.h = hiddenstate
            self.prevNode = previousNode
            self.wordid = wordId
            self.logp = logProb
            self.leng = length

        def eval(self, alpha=1.0):
            reward = 0
	    # Add here a function for shaping a reward
            return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # penalty parameter

        def __lt__(self, other):
            return self.leng < other.leng  # for conflict

        def __gt__(self, other):
            return self.leng > other.leng        
    INPUT_DIM =36
    ENCODER_INPUT_DIM = 2
    HID_DIM = 128
    N_LAYERS = 2
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 16
    EARLY_STOP_SIGN = False

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM , HID_DIM, N_LAYERS)
    dec = Decoder(ENCODER_INPUT_DIM,DEC_EMB_DIM , HID_DIM, N_LAYERS)

    model = Seq2Seq(enc, dec, device).to(device)
    patience = 7	
    early_stopping = EarlyStopping(patience, verbose=True)	

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.15, 0.15)
    #         nn.init.orthogonal_(param.data)

    model.apply(init_weights)


    # In[ ]:


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')


    # In[ ]:


    optimizer = optim.Adam(model.parameters(),weight_decay=0.00001,lr=0.01)
    criterion = nn.MSELoss()
    
    def train(model, dataloader,optimizer, criterion, clip):
        global firstinput
        model.train()

        epoch_loss = 0

        for x,y in dataloader:

            x=x.transpose(1,0)
            y=y.transpose(1,0)
            x=x.to('cuda')
            y=y.to('cuda')
            firstinput=y[0,:,:]
            y=y[1:,:,:]
            optimizer.zero_grad()

            output = model(x, y)
            output = output.to('cuda')


    #         loss = criterion(output, y)
            #print(output.size())
            loss = 3*criterion(output[:,:,1],y[:,:,1])+criterion(output[:,:,0],y[:,:,0])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
            #print(epoch_loss)

        return epoch_loss/len(dataloader)


    # In[ ]:


    def evaluate(model, validataloader, criterion):

        model.eval()

        epoch_loss = 0

        with torch.no_grad():

            for x,y in validataloader:

                x=x.transpose(1,0)
                y=y.transpose(1,0)
                x=x.to('cuda')
                y=y.to('cuda')
                firstinput=y[0,:,:]
                y=y[1:,:,:]
                optimizer.zero_grad()

                output = model(x, y, 0) #turn off teacher forcing
                output = output.to('cuda')


                loss = 3*criterion(output[:,:,1],y[:,:,1])+criterion(output[:,:,0],y[:,:,0])
                early_stopping(loss, model)
                if early_stopping.early_stop:
                    EARLY_STOP_SIGN = True
                epoch_loss += loss.item()


        return epoch_loss / len(validataloader)


    # In[ ]:


    def test(model, testdataloader, criterion):
        global j
        global firstinput
        global test_result
        model.eval()

        epoch_loss = 0

        with torch.no_grad():

            for x,y in testdataloader:

                x=x.transpose(1,0)
                y=y.transpose(1,0)
                x=x.to('cuda')
                y=y.to('cuda')
                firstinput=y[0,:,:]
                y=y[1:,:,:]
                optimizer.zero_grad()

                output = model(x, y, 0) #turn off teacher forcing
                test_result[:,j:j+BATCH_SIZE,:]=output
                j=j+BATCH_SIZE
                output = output.to('cuda')


    #             loss = criterion(output, y)
                loss = 3*criterion(output[:,:,1],y[:,:,1])+criterion(output[:,:,0],y[:,:,0])
                epoch_loss += loss.item()

    #     print(len(testdataloader))

        return epoch_loss / len(testdataloader)


    # In[ ]:


    N_EPOCHS = 40
    CLIP = 1
    global test_result
    test_result=np.zeros([25,80000,2])
    pynvml.nvmlInit()
    handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('this is seq2seq with beam search\n')
    for epoch in range(N_EPOCHS):
        global j
        j=0
        start_time = time.process_time()
        train_loss = train(model, trainloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valiloader, criterion)
        end_time = time.process_time()
        print(f'Epoch: {epoch+1:02} | Time: {end_time-start_time}s')
        print(f'\tTrain Loss: {train_loss:.3f} |  Val. Loss: {valid_loss:.3f}')
        #writer.add_scalars('loss',{'train_loss': train_loss,
                                   #'valid_loss': valid_loss},epoch )
        test_loss = test(model, testloader, criterion)
        if test_loss<3 or EARLY_STOP_SIGN:
            print('testloss:',test_loss)
            print('Early Stopping!')
            print(y_test.shape)
            test_result_record=test_result[:,:j,:]
            print(test_result_record.shape)
            np.save(r'./result/seq2seq_beam_search_predict_tra.npy',test_result_record)
            np.save(r'./result/true_tra.npy',y_test[:,1:,:])
            break
        if epoch == 39:
            print('testloss:',test_loss)
            test_result_record=test_result[:,:j,:]
            np.save(r'./result/seq2seq_beam_search_predict_tra.npy',test_result_record)
            np.save(r'./result/true_tra.npy',y_test[:,1:,:])
            break
    print('meminfo.used:',meminfo.used/(1024*1024))
    print('meminfo.total:',meminfo.total/(1024*1024))
    
    return 0

