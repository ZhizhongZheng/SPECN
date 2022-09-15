# SPECN
The code and data of SPECN.

# Requirements
Python  3

PyTorch 1.10.2

Numpy   1.18.1

SciPy   1.5.4

# Usage
run python train_caser.py

# data
user item rating
Where all ratings are 1.

#Comments
Movielens-1M
d=50, L=5, T=3
 merge_out_vu = self.ac_fc(torch.add(torch.matmul(out_v, self.CW00), out_h))
 user_merge0 = self.ac_fc(torch.add(torch.matmul(out_u, self.CW11), out_u1))
