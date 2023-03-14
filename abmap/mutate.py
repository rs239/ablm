import random

all_aas = ['A', 'G', 'V', 'I', 'L', 'F', 'P', 'Y', 'M', 'T',
           'S', 'H', 'N', 'Q', 'W', 'R', 'K', 'D', 'E', 'C']

aa_groupings1 = [['A', 'G', 'V'],
                ['I', 'L', 'F', 'P'],
                ['Y', 'M', 'T', 'S'],
                ['H', 'N', 'Q', 'W'],
                ['R', 'K'],
                ['D', 'E'],
                ['C'],
                ['X']]

aa_groupings2 = [['A', 'G', 'V', 'I', 'L', 'F', 'P', 'Y', 'M', 'T',
                  'S', 'H', 'N', 'Q', 'W', 'R', 'K', 'D', 'E', 'C'],
                 ['X']]

def dict_from_group(grouping):
    mutate_dict = dict()

    for group in grouping:
        for aa in group:
            gc = group.copy()
            if len(gc) > 1:
                gc.remove(aa)
            mutate_dict[aa] = gc

    return mutate_dict


mutate_cat1 = dict_from_group(aa_groupings1)
mutate_cat2 = dict_from_group(aa_groupings2)


def generate_mutation(sequence, mask, p=0.5, mut_type = "cat1"):
    '''
    Increasing p yields more mutations. Consider using a higher p when k is lower
    '''

    if mut_type == "cat1":
        mutate_dict = mutate_cat1
    elif mut_type == "cat2":
        mutate_dict = mutate_cat2

    seq = list(sequence)

    # mutation happens here:
    for i, mask_bool in enumerate(mask):
        rand_num = random.uniform(0, 1)
        if mask_bool and rand_num >= p: 
            seq[i] = random.choice(mutate_dict[seq[i]])

    mutated_seq = "".join(seq)
    return mutated_seq
