#!/usr/bin/env python
# coding: utf-8

# # ESM-IF

# In[1]:


import pandas
import numpy as np
from copy import deepcopy


# In[2]:


## Verify that pytorch-geometric is correctly installed
## if notebook crashes it is for incompatible versions of torch and torch_geometric
import torch_geometric

# import torch_sparse
from torch_geometric.nn import MessagePassing


# # Loading mutation data and group them

# In[3]:


from dataclasses import dataclass


@dataclass
class mutation_data:
    pdb_code: str = None
    chain: str = None
    position: int = None
    from_aa: str = None
    to_aa: str = None
    ddg: float = None

    def parse_mut(self, mut_string):
        _raw3 = mut_string.strip().split()
        assert len(_raw3) == 3
        assert len(_raw3[0]) == 5
        self.chain = _raw3[0][-1].upper()
        self.pdb_code = _raw3[0][:-1].lower()
        self.from_aa = _raw3[1][0].upper()
        self.to_aa = _raw3[1][-1].upper()
        self.position = int(_raw3[1][1:-1])
        self.ddg = float(_raw3[2])
        return self

    def to_mut_line(self):
        return " ".join(
            [
                "".join([self.pdb_code, self.chain]),
                "".join(
                    [
                        self.from_aa,
                        str(self.position),
                        self.to_aa,
                    ]
                ),
                str(self.ddg),
            ]
        )


# In[4]:


# _test_mut='5dfrA W30N  -2.07'

# z = mutation_data()
# z.parse_mut(_test_mut)
# z.to_mut_line()
# for _a, _b in zip(
#     _test_mut.split(), mutation_data().parse_mut(_test_mut).to_mut_line().split()
# ):
#     # print( _a == _b)
#     assert _a == _b
# _test_mut.split(), z.to_mut_line().split()


# In[5]:


mutations = []
errors = []

with open("Datasets_2024_05_30/Intersection_Varibench_S2648.mut") as _f:
    for line in _f:
        try:
            _ = mutation_data().parse_mut(line)
            mutations.append(_)
        except:
            errors.append(line)
# len(mutations), len(errors)


# In[6]:


# errors


# In[7]:


mutations_df = pandas.DataFrame(mutations)
# mutations_df.head()


# In[8]:


# mutations_df.head() #pivot(['pdb_code','chain'])['position']
zegroups = mutations_df.groupby(["pdb_code", "chain", "position"]).groups


# In[9]:


# _grpvals= [('_'.join( map(str,_)), zegroups[_]) for _ in zegroups ]
_grpvals = [list(_) + [zegroups[_]] for _ in zegroups]
# _grp=_grpvals[0]
# mutations_df.iloc[_grpvals[0]]


# In[ ]:


# ## eval esm-if

# In[10]:


pdb_dir = "Intersection_Vari_S2648_pdbs/"
out_dir = "esmif_out/"
RT = 1.987 * 298

def get_res_ids(structure):
    return [ _[1]  
            for _ in zip(structure.get_annotation("atom_name"), structure.get_annotation("res_id")) 
            if _[0] == 'CA' ]

def mutate_labelled_seq(lseq, position, new_aa_symbol):
    "mutate the string representing the protein sequence"
    assert type(lseq) == type(dict())
    lseq[position] = new_aa_symbol
    return ''.join(lseq.values())
    #return "".join([seq[: position - 1], new_aa_symbol.capitalize(), seq[(position):]])

def get_aa(lseq, pos):
    "get aminoacid at position in labelled sequence"
    assert type(lseq) == type(dict())
    return lseq[pos]




# In[11]:


import esm
import esm.inverse_folding as esmif
from copy import deepcopy

model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()


# In[ ]:


for _grp in _grpvals:
    print(_grp)
    # try loading a pdb
    pdbpath = pdb_dir + ".".join([_grp[0], "pdb"])  # .pdb format is also acceptable
    chain_id = _grp[1]
    structure = esmif.util.load_structure(pdbpath, chain_id)
    coords, native_seq = esmif.util.extract_coords_from_structure(structure)
    print("Native sequence:")
    print(native_seq)

    ca_labelz = get_res_ids(structure)
    assert len(ca_labelz) ==  len(native_seq)
    _native_label_seq =dict( _ for _ in zip(ca_labelz, native_seq))

    # In[13]:


    _esmif_df = deepcopy(mutations_df.iloc[_grp[3]])


    _native_aa = get_aa(_native_label_seq, _grp[2])
    assert _native_aa == _esmif_df.loc[_esmif_df.index[0]]["from_aa"]


    # In[24]:


    # score native
    ll_native, ll__native_withcoor = esmif.util.score_sequence(
        model, alphabet, coords, native_seq
    )

    print(
        f"average log-likelihood on entire sequence: {ll_native:.4f} (perplexity {np.exp(-ll_native):.4f})"
    )

    _esmif_df["ll_native"] = ll_native




    _seqs = []
    for _ in _esmif_df["to_aa"]:
        assert _ != _native_aa
        _seqs.append(mutate_labelled_seq(_native_label_seq, _grp[2], _))
    # _seqs


    # In[27]:


    # score mutated seqs
    _ll_muts = []
    for _ in _seqs:
        _ll, _ll_withcoor = esmif.util.score_sequence(model, alphabet, coords, _)
        print(
            f"average log-likelihood on entire sequence: {_ll:.4f} (perplexity {np.exp(-_ll):.4f})"
        )
        _ll_muts.append(_ll)
    # _ll_muts
    # print(f'average log-likelihood excluding missing coordinates: {ll_withcoord:.2f} (perplexity {np.exp(-ll_withcoord):.2f})')


    # In[28]:


    _esmif_df["ll_mut"] = _ll_muts
    # _esmif_df


    # In[29]:


    _esmif_df["delta"] = _esmif_df["ll_mut"] - _esmif_df["ll_native"]
    _esmif_df["guess"] = RT * _esmif_df["delta"]

    # In[44]:


    outpath = out_dir + "_".join(map(str, _grp[:3])) + ".csv"
    _esmif_df.to_csv(outpath)
