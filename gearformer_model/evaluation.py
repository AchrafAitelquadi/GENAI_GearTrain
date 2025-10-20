import torch
from utils.dataset import load_data
from decimal import Decimal as D
from decimal import getcontext
import csv
from utils.helper import is_grammatically_correct, is_physically_feasible
from utils.dataset import load_data, MyDataset
from tqdm import tqdm
from models.load_model import loading_model
import os
getcontext().prec = 5
from utils.config import config
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


class Eval(load_data):
    def __init__(self, args, output_size, model, encoder, decoder):
        super(Eval, self).__init__(args)
        self.args = args
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.cuda().eval()
        self.decoder.cuda().eval()
        self.output_size = output_size

    def get_output_sequence_accuracy(self, x_val, y_val, csvwriter):
        correct = 0
        valid = 0
        grammar = 0
        
        input_vec = x_val.clone().detach().to(torch.float32).cuda()
        decoder_input = torch.zeros(self.output_size).cuda()
        decoder_input[0] = 1
        decoder_input = decoder_input.repeat(input_vec.shape[0], 1).cuda()

        with torch.no_grad():
                
            encoded_input = self.encoder(input_vec)
            prompt = torch.zeros((len(y_val),1)).cuda()
            all_out = self.decoder.generate(prompts=prompt, context=encoded_input, seq_len=20)

        for inx in range(len(all_out)):
            out = all_out[inx]

            out = list(map(self.inx2name, out.cpu().tolist()))

            out.append("<end>")
            target_inx = out.index("<end>")
            out = out[:target_inx+1]


            csvwriter.writerow([input_vec[inx][0],input_vec[inx][1], input_vec[inx][2], input_vec[inx][3], input_vec[inx][4], input_vec[inx][5], input_vec[inx][6], input_vec[inx][7], out])


            if is_grammatically_correct(self.args, ['<start>'] + out):
                grammar += 1
                if is_physically_feasible(['<start>'] + out, self.args.catalogue_path):
                    valid += 1
        return correct , valid, grammar


    def accuracy(self, seq1, seq2):
        seq2 = seq2[1:]
        for i in range(len(seq1)):
            if int(seq1[i]) != int(seq2[i]):
                return False
            if seq1[i] == 27:
                break
        return True