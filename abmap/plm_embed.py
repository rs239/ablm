import torch
import abmap.base_config as base_config
from dscript.alphabets import Uniprot21
import re



def reload_models_to_device(device_num=0, plm_type='beplerberger'):
    base_config.device = "cuda:{}".format(device_num)
    device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    print(f'{plm_type} loaded to {device}')

    ######### Bepler & Berger ########
    if plm_type == 'beplerberger':
        from dscript import pretrained as ds_pre
        global bb_model
        bb_model = ds_pre.get_pretrained("lm_v1")
        bb_model = bb_model.to(device)
        bb_model.eval()

    ######### ESM-1b or ESM-2 ##########
    elif plm_type in ('esm1b', 'esm2'):
        global esm_model, esm_batch_converter
        if plm_type == 'esm1b':
            esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
        else:
            esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        esm_batch_converter = esm_alphabet.get_batch_converter()
        esm_model = esm_model.to(device)
        esm_model.eval()
    
    ######### TAPE ##########
    elif plm_type == 'tape':
        global tape_model, tape_tokenizer
        from tape import ProteinBertModel, TAPETokenizer
        tape_model = ProteinBertModel.from_pretrained('bert-base')
        tape_tokenizer = TAPETokenizer(vocab='iupac')
        tape_model = tape_model.to(device)
        tape_model.eval()

    ######### ProtBert #########
    elif plm_type == 'protbert':
        from transformers import BertModel, BertTokenizer
        global protbert_tokenizer, protbert_model
        protbert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        protbert_model = BertModel.from_pretrained("Rostlab/prot_bert")
        protbert_model = protbert_model.to(device)
        protbert_model.eval()

    else:
        raise ModuleNotFoundError(f"PLM {plm_type} not found...")


def embed_sequence(sequence, embed_type = "beplerberger", embed_device = None, embed_model=None):
    if embed_device is None:
        device = torch.device(base_config.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(embed_device)

    if embed_type == "beplerberger":
        from dscript.alphabets import Uniprot21
        # print("using Bepler & Berger's Embedding...")

        with torch.no_grad():
            alphabet = Uniprot21()
            es = torch.from_numpy(alphabet.encode(sequence.encode('utf-8')))
            x = es.long().unsqueeze(0)
            x = x.to(device)

            if embed_model is None:
                z = bb_model.transform(x)
            else:
                z = embed_model.transform(x)

            return z.detach().cpu()[0]


    elif embed_type == "esm1b" or embed_type == "esm2":
        # print("using FAIR's esm-1b...")

        data = [
            ("seq", sequence),
        ]
        batch_labels, batch_strs, batch_tokens = esm_batch_converter(data)
        # think it adds beginning and end tokens in the conversion process: ex) seq length 16 --> 18

        batch_tokens = batch_tokens.to(device)
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
        model = tape_model.eval().to(device)

        token_ids = torch.tensor([tape_tokenizer.encode(sequence)]) #.to(device)
        # token_ids = token_ids.to()
        with torch.no_grad():
            if embed_model is None:
                output = model(token_ids.to(device))
            else:
                output = embed_model(token_ids.to(device))
        embedding = output[0][0,1:-1,:] # remove start and end tokens

        return embedding


    elif embed_type == "dscript":
        from dscript.alphabets import Uniprot21
        assert embed_device is None and embed_model is None # not implemented yet
        # print("using D-Script Embedding (Bepler&Berger + projection)...")
        with torch.no_grad():
            alphabet = Uniprot21()
            es = torch.from_numpy(alphabet.encode(sequence.encode('utf-8')))
            x = es.long().unsqueeze(0)
            x = x.to(device)
            z = bb_model.transform(x)
            # return z.cpu()[0]

        model = ds_model.eval().to(device)
        embedding = model.embedding(z)[0]

        return embedding #.to()

    elif embed_type == 'protbert':

        sequence_spaced = " ".join(sequence)
    
        with torch.no_grad():
            encoded_input = protbert_tokenizer(sequence_spaced, return_tensors='pt')
            # encoded_input = encoded_input.to(device)
            # print(encoded_input)
            for key in encoded_input.keys():
                encoded_input[key] = encoded_input[key].to(device)
            output = protbert_model(**encoded_input)
            feature = output.last_hidden_state[0][1:-1,:]

        return feature

