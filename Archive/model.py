#!git lfs install
#!git clone https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english

from collections import defaultdict

d_int = defaultdict(int)

d_int['hello']

d_int

d_int['hoo'] += 1
names_dict = {k:v for k, v in sorted(d_int.items(), key=lambda i : i[1], reverse=True)}

assert names_dict['sfs'] == 0

defaultdict(int, names_dict)
