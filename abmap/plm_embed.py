import torch
import base_config

from dscript import language_model as lm
from dscript import pretrained as ds_pre
from dscript.alphabets import Uniprot21
import re



def reload_models_to_device(device_num=0, plm_type='beplerberger'):
    base_config.device = "cuda:{}".format(device_num)
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    print(device)

    ######### Bepler & Berger ########
    if plm_type == 'beplerberger':
        global bb_model
        bb_model = ds_pre.get_pretrained("lm_v1")
        bb_model = bb_model.cuda(device)
        bb_model.eval()

    ######### ESM-1b ##########
    elif plm_type == 'esm1b':
        global esm_model, esm_batch_converter
        esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
        esm_batch_converter = esm_alphabet.get_batch_converter()
        esm_model = esm_model.cuda(device)
        esm_model.eval()

    ######### TAPE ##########
    elif plm_type == 'tape':
        global tape_model, tape_tokenizer
        from tape import ProteinBertModel, TAPETokenizer
        tape_model = ProteinBertModel.from_pretrained('bert-base')
        tape_tokenizer = TAPETokenizer(vocab='iupac')
        tape_model = tape_model.cuda(device)
        tape_model.eval()

    ######### ProtBert #########
    elif plm_type == 'protbert':
        from transformers import BertModel, BertTokenizer
        global protbert_tokenizer, protbert_model
        protbert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        protbert_model = BertModel.from_pretrained("Rostlab/prot_bert")
        protbert_model = protbert_model.cuda(device)
        protbert_model.eval()

    else:
        raise ModuleNotFoundError(f"PLM {plm_type} not found...")


def embed_sequence(sequence, embed_type = "beplerberger", embed_device = None, embed_model=None):
    if embed_device is None:
        device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(embed_device)

    if embed_type == "beplerberger":
        # print("using Bepler & Berger's Embedding...")

        with torch.no_grad():
            alphabet = Uniprot21()
            es = torch.from_numpy(alphabet.encode(sequence.encode('utf-8')))
            x = es.long().unsqueeze(0)
            x = x.cuda(device)

            if embed_model is None:
                z = bb_model.transform(x)
            else:
                z = embed_model.transform(x)

            return z.detach().cpu()[0]


    elif embed_type == "esm1b":
        # print("using FAIR's esm-1b...")

        data = [
            ("seq", sequence),
        ]
        batch_labels, batch_strs, batch_tokens = esm_batch_converter(data)
        # think it adds beginning and end tokens in the conversion process: ex) seq length 16 --> 18

        batch_tokens = batch_tokens.cuda(device)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            if embed_model is None:
                results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
            else:
                results = embed_model(batch_tokens, repr_layers=[33], return_contacts=True)

        out_temp = results["representations"][33]
        embedding = out_temp[0,1:-1,:] # remove start and end tokens

        return embedding #(out_temp[0], out_temp[1])


    elif embed_type == "tape":
        assert embed_device is None and embed_model is None # not implemented yet
        # print("using Berkeley's TAPE...")
        model = tape_model.eval().cuda(device)

        token_ids = torch.tensor([tape_tokenizer.encode(sequence)]) #.cuda(device)
        # token_ids = token_ids.cuda()
        with torch.no_grad():
            if embed_model is None:
                output = model(token_ids.cuda(device))
            else:
                output = embed_model(token_ids.cuda(device))
        embedding = output[0][0,1:-1,:] # remove start and end tokens

        return embedding


    elif embed_type == "dscript":
        assert embed_device is None and embed_model is None # not implemented yet
        # print("using D-Script Embedding (Bepler&Berger + projection)...")
        with torch.no_grad():
            alphabet = Uniprot21()
            es = torch.from_numpy(alphabet.encode(sequence.encode('utf-8')))
            x = es.long().unsqueeze(0)
            x = x.cuda(device)
            z = bb_model.transform(x)
            # return z.cpu()[0]

        model = ds_model.eval().cuda(device)
        embedding = model.embedding(z)[0]

        return embedding #.cuda()

    elif embed_type == 'protbert':

        sequence_spaced = " ".join(sequence)
    
        with torch.no_grad():
            encoded_input = protbert_tokenizer(sequence_spaced, return_tensors='pt')
            # encoded_input = encoded_input.cuda(device)
            # print(encoded_input)
            for key in encoded_input.keys():
                encoded_input[key] = encoded_input[key].cuda(device)
            output = protbert_model(**encoded_input)
            feature = output.last_hidden_state[0][1:-1,:]

        return feature

