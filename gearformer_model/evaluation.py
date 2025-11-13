import torch
from utils.data_handle import load_data
from decimal import Decimal as D
from decimal import getcontext
import csv
from utils.helper import is_grammatically_correct, is_physically_feasible
from utils.data_handle import load_data, MyDataset
from tqdm import tqdm
from models.load_model import loading_model
import os
getcontext().prec = 5
from utils.config_file import config
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
        valid = 0
        grammar = 0
        
        input_vec = x_val.to(torch.float32).cuda()

        with torch.no_grad():
                
            encoded_input = self.encoder(input_vec)
            # prompt = token <start> = index 0
            prompt = torch.zeros((len(y_val),1)).cuda()
            all_out = self.decoder.generate(prompts=prompt, context=encoded_input, seq_len=20)

        for inx in range(len(all_out)):
            out_inx = all_out[inx].cpu().tolist()
            out_tok = list(map(self.inx2name, out_inx))

            # Couper au token <end>
            if "<end>" in out_tok:
                end_pos = out_tok.index("<end>")
                out_tok = out_tok[:end_pos + 1]

            # Écrire dans CSV : 8 features + séquence générée
            row = [float(x.item()) for x in input_vec[inx][:8]]
            row.append(out_tok)
            csvwriter.writerow(row)


            # Vérif grammaticale
            seq = ['<start>'] + out_tok
            if is_grammatically_correct(self.args, seq):
                grammar += 1

                # Vérif physique
                if is_physically_feasible(seq, self.args.catalogue_path):
                    valid += 1

        return valid, grammar



        
if __name__ == "__main__":
    args = config()
    max_length = 20
    output_size = 53 # number of classes
    with_weight = False
    input_size = 8

    csv_file_name = str(args.model_name)+"_EPOCH"+str(args.epoch)+"_BS"+str(args.BS)+"_WWL"+str(args.WWL)+"_lr"+str(args.lr)+".csv"
    csvfile = open(csv_file_name, 'w', newline='')
    csvwriter = csv.writer(csvfile)

    encoder, decoder = loading_model(args, input_size, output_size, max_length)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_chackpoint_name)))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_chackpoint_name)))

    """ ### This is to calculate the number of parameters:
    pytorch_total_params = sum(p.numel() for p in encoder.parameters())
    pytorch_total_params_t = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print("encoder:", pytorch_total_params, pytorch_total_params_t)
    pytorch_total_params = sum(p.numel() for p in decoder.parameters())
    pytorch_total_params_t = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("decoder:", pytorch_total_params, pytorch_total_params_t) """
    
    encoder.cuda().eval()
    decoder.cuda().eval()

    eval = Eval(args, output_size, args.model_name, encoder, decoder)

    x_val, y_val, target_length, weight_val, _ = eval.get_all_data(True, with_weight)


    val_loader = torch.utils.data.DataLoader(MyDataset(x_val, y_val, target_length, weight_val), batch_size=args.BS, shuffle=False, num_workers=0)
    all_valid, all_grammar = 0, 0
    dataset_length = len(y_val)

    print("dataset_length:", len(y_val))

    for x_val, y_val, target_length, _ in tqdm(val_loader):
        valid, grammar = eval.get_output_sequence_accuracy(x_val, y_val, csvwriter)
        all_valid += valid
        all_grammar += grammar
    
    print(
        "valid_ratio:", all_valid / dataset_length,
        "grammar_ratio:", all_grammar / dataset_length
    )