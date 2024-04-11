from abmap.modified_anarci.anarci.anarci import run_anarci


def get_anarci_pos(seq, return_chain_type=False, scheme="c"):
    anarci_out = run_anarci(seq, scheme="c")
    anarci_numbering = anarci_out[1][0][0][0]
    index = []
    seq_reconst = ""
    # ignore "-"
    for element in anarci_numbering:
        idx = str(element[0][0])
        if element[0][1] != " ":
            idx += element[0][1]  # (1, 'A') -> '1A'
        aa = element[1]  # 'Q' or '-'
        if aa != "-":
            index.append(idx)
            seq_reconst += aa

    # add constant region
    const_parts = seq.split(seq_reconst)
    index = (
        ["C" for i in range(len(const_parts[0]))]
        + index
        + ["C" for i in range(len(const_parts[1]))]
    )

    if return_chain_type:
        return index, anarci_out[2][0][0]['chain_type']
    return index
