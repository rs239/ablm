from protein_embedding import ProteinEmbedding
from model import AbMAPAttn
from embed import reload_models_to_device
import torch
from torch import nn

reload_models_to_device(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

demo_seq = 'EVKLQESGGDLVQPGGSLKLSCAASGFTFSSYTMSWVRQTPEKRLEWVASINNGGGRTYYPDTVKGRFTISRDNAKNTLYLQMSSLKSEDTAMYYCVRHEYYYAMDYWGQGTTVTVSSA'

prot_embed = ProteinEmbedding(sequence = demo_seq, chain_type='H')

print("Sequence:\n{}".format(demo_seq))
print("Sequence length:", len(demo_seq))
print()

# embed sequence according to a specific type:
prot_embed.embed_seq(embed_type = 'beplerberger')
print("Embedding shape:", prot_embed.embedding.shape)
print()

# create a cdr mask of the sequence: (1, 2, 3)'s for CDR regions, 0's for else
prot_embed.create_cdr_mask()
print("CDR Mask:")
print(demo_seq)
print("".join(map(str, map(int, prot_embed.cdr_mask.tolist()))))
print()

# create a CDR embedding with no mutation adjustments:
cdr_embed_nomut = prot_embed.create_cdr_embedding_nomut()
print("CDR Embedding with no mutation adjustments:", cdr_embed_nomut.shape)

# create CDR embedding with mutation adjustments with/without separators:
cdr_embed_sep = prot_embed.create_cdr_specific_embedding(embed_type='beplerberger', k=100, seperator=True, mask=False)
print("CDR Embedding with SEP token shape:", cdr_embed_sep.shape)
cdr_embed_nosep = prot_embed.create_cdr_specific_embedding(embed_type='beplerberger', k=100, seperator=False, mask=False)
print("CDR Embedding without SEP token shape:", cdr_embed_nosep.shape)
print()

print("(from here onwards) default is separator token = False")

# create CDR embedding with mutation adjustments with/without CDR masks:
cdr_embed_withmask = prot_embed.create_cdr_specific_embedding(embed_type='beplerberger', k=100, mask=True)
print("CDR Embedding with CDR mask shape:", cdr_embed_withmask.shape)
cdr_embed_nomask = prot_embed.create_cdr_specific_embedding(embed_type='beplerberger', k=100, mask=False)
print("CDR Embedding without CDR mask shape:", cdr_embed_nomask.shape)
print()

print("(from here onwards) default is mask = True")

# Load pre-trained AbMAP (Bepler & Berger) Model
model = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512, 
                   proj_dim=252, num_enc_layers=1, num_heads=16).to(device)
pretrained_path = 'models/AbMAP_H_beplerberger_epoch50.pt'
checkpoint = torch.load(pretrained_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Variable and Fixed-Length Embeddings:
feat_H = torch.unsqueeze(cdr_embed_withmask, dim=0).to(device)
task = 0 # structure

with torch.no_grad():
    var_len_feat, _ = model(feat_H, feat_H, None, None, task=task, return3=True)

with torch.no_grad():
    fix_len_feat, _ = model(feat_H, feat_H, None, None, task=task, task_specific=True)

print("Variable Length Feature shape:", var_len_feat.shape)
print("Fixed Length Feature shape:", fix_len_feat.shape)