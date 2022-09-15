# SPECN
The code and data of SPECN.

# Requirements
Python  3

PyTorch 1.10.2

Numpy   1.18.1

SciPy   1.5.4

# Usage
run python train_specn.py

# data
user item rating

Where all ratings are 1.

# Comments
Movielens-1M：

d=80, L=5, T=3

merge_out_vu = self.ac_fc(torch.add(torch.matmul(out_v, self.CW00), out_h))

user_merge0 = self.ac_fc(torch.add(torch.matmul(out_u, self.CW11), out_u1))

Gowalla and CDs：

d=150, L=3, T=3;d=350, L=3, T=2

merge_out_vu = torch.sigmoid(torch.add(torch.matmul(out_v, self.CW00), out_h))

user_merge0 = torch.sigmoid(torch.add(torch.matmul(out_u, self.CW11), out_u1))

The numbers of capsules will change when L changes, please read the code.

# Acknowledgment

This project is heavily built on [Spotlight](https://github.com/maciejkula/spotlight), [Jiaxi Tang](https://github.com/graytowne) and [Maciej Kula](https://github.com/maciejkula). Thanks!
