import torch
import torch.nn as nn
from utils import activation_getter
from transformer.Modules import ScaledDotProductAttention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SPECN(nn.Module):
    """
    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, model_args):
        super(SPECN, self).__init__()
        self.args = model_args
        L = self.args.L
        T = self.args.T
        dims = self.args.d

        self.n_h = self.args.nh
        self.n_v = self.args.nv

        self.n_u = self.args.nu
        self.n_u1 = self.args.nu1


        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)


    #################### item ###########################
        # vertical conv layer
        self.conv_v2 = nn.Conv2d(1, self.n_v,( 2 * L , 1))
        # horizontal conv layer
        self.conv_h2 = nn.Conv2d(1, self.n_h, (2, dims))
    ################### item ############################


    #################### user ###########################
        # vertical conv layer
        self.conv_u = nn.Conv2d(1, self.n_u, (1, 1))
        # horizontal conv layer
        self.conv_u1 = nn.Conv2d(1, self.n_u1, (1, dims))
    #################### user ###########################


        # 维度计算
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * 9
        # 维度计算
        self.fc1_dim_u = self.n_u * dims
        self.fc1_dim_u1 = self.n_u1 * 1


        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, self.fc1_dim_h + dims + self.n_u1)
        self.b2 = nn.Embedding(num_items, 1)

        #胶囊网络和特征融合矩阵
        self.CW1 = torch.nn.Parameter(torch.randn([self.args.nh, self.args.d]), requires_grad=True).to(device)
        self.CW2 = torch.nn.Parameter(torch.randn([self.args.d, self.args.nv]), requires_grad=True).to(device)
        self.CW00 = torch.nn.Parameter(torch.randn([self.fc1_dim_v, self.fc1_dim_h]), requires_grad=True).to(device)
        self.CW11 = torch.nn.Parameter(torch.randn([self.fc1_dim_u, self.fc1_dim_u1]), requires_grad=True).to(device)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()
        self.cache_x = None

    def forward(self, seq_var, user_var, item_var, for_pred = False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).
        Parameters
        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up and self-attention
        item_embs = self.item_embeddings(seq_var)
        user_emb = self.user_embeddings(user_var)
        user_emb1 = user_emb.squeeze(1)
        user_emb0 = user_emb.unsqueeze(1)

        #添加用户注意特征的嵌入矩阵
        #############################################################################################################
        embed_list_final = []
        for m in range(item_embs.shape[0]):
            embed_list = []
            for n in range(item_embs.shape[1]):
                items_alone = item_embs[m][n].unsqueeze(0)
                user_alone = user_emb1[m].unsqueeze(0)
                ScaledDotProductAttention0 = ScaledDotProductAttention(temperature=torch.sqrt(torch.tensor(self.args.d)))
                user_alone_selfattention = ScaledDotProductAttention0(q = user_alone, k = user_alone, v = items_alone )
                merge_alone = torch.cat([items_alone, user_alone_selfattention], dim=0)
                embed_list.append(merge_alone)
            item_embs0 = torch.cat(embed_list, dim=0)
            embed_list_final.append(item_embs0)
        item_embs_final = (torch.stack(embed_list_final, dim=0)).unsqueeze(1)
        #############################################################################################################

        #胶囊网络层
        #############################################################################################################
        out_h, out_v, out_u, out_u1 = None, None, None, None
        # 垂直胶囊网络
        if self.n_v:
            out_v = torch.sigmoid(self.conv_v2(item_embs_final).squeeze(-2))
            # print(out_v.shape)
            out_v = self.item_routing(inputs=out_v, CW=self.CW2, num_capsule=4, dim_capsule=self.args.d,
                                      input_num_capsule=4, input_dim_capsule=self.args.nv, routings=3)
            out_v = out_v.view(-1, self.fc1_dim_v)
            out_v = self.dropout(out_v)
        # 水平胶囊网络
        if self.n_h:
            out_h = torch.sigmoid(self.conv_h2(item_embs_final).squeeze(-1))
            out_h = self.item_routing1(inputs=out_h, CW=self.CW1, num_capsule=9, dim_capsule=self.args.d,
                                       input_num_capsule=9, input_dim_capsule=self.args.nh, routings=3)
            out_h = out_h.view(-1, self.fc1_dim_h)
            out_h = self.dropout(out_h)
        #############################################################################################################

    #用户更细粒度特征提取
    #############################################################################################################
        if self.n_u:
            out_u = torch.sigmoid(self.conv_u(user_emb0).squeeze(-2))
            out_u = out_u.contiguous().view(-1, self.fc1_dim_u)
            out_u = self.dropout(out_u)
        if self.n_u1:
            out_u1 = torch.sigmoid(self.conv_u1(user_emb0).squeeze(-2))
            out_u1 = out_u1.contiguous().view(-1, self.fc1_dim_u1)
            out_u1 = self.dropout(out_u1)
    #############################################################################################################

    #特征融合！！！
    ################################################# item #############################################################
        merge_out_vu = self.ac_fc(torch.add(torch.matmul(out_v, self.CW00), out_h))
        user_merge0 = self.ac_fc(torch.add(torch.matmul(out_u, self.CW11), out_u1))
        user_merge = torch.cat([user_emb1, user_merge0], 1)  # 维度是 dims + self.n_u1
        x = torch.cat([merge_out_vu,user_merge], 1)
    ################################################# item #############################################################

        w2 = self.W2(item_var)
        b2 = self.b2(item_var)
        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()
        return res

    #胶囊网络
    #############################################################################################################
    def squash_v1(self, x, dim=-1):
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm)
    def softmax(self, input, dim):
        exp = torch.exp(input)
        return exp / torch.sum(exp, dim, keepdim=True)
    def item_routing(self, inputs, CW,num_capsule, dim_capsule, input_num_capsule, input_dim_capsule,routings):
        inputs=torch.transpose(inputs,1,2)
        inputs_expand = torch.unsqueeze(inputs, 1)
        CW = torch.unsqueeze(torch.unsqueeze(CW, 0), -1).repeat(num_capsule, 1, 1, input_num_capsule)
        inputs_tiled = inputs_expand.repeat([1,num_capsule, 1,1])
        define_list=[]
        for i in range(len(inputs_tiled)):
           define_list.append(torch.reshape((torch.matmul(CW, torch.reshape(inputs_tiled[i], (num_capsule,dim_capsule,input_num_capsule,1)))),(num_capsule,dim_capsule ,input_dim_capsule)))
        inputs_hat=(torch.stack(define_list))
        b = torch.zeros([inputs_hat.shape[0], num_capsule, input_dim_capsule]).to(device)
        assert routings > 0, 'The routings should be > 0.'
        for i in range(routings):
            c = self.softmax(b, dim=1)
            c = torch.reshape(c,(inputs_hat.shape[0], num_capsule, input_dim_capsule, 1))
            outputs = self.squash_v1(torch.reshape(torch.matmul(inputs_hat, c),(inputs_hat.shape[0], num_capsule,dim_capsule)))
            if i < routings - 1:
                b += torch.reshape(torch.matmul(torch.transpose(inputs_hat,2,3), torch.reshape(outputs, (inputs_hat.shape[0], num_capsule,dim_capsule, 1))),
                                  (inputs_hat.shape[0], num_capsule, input_dim_capsule))
        return outputs
    def item_routing1(self, inputs, CW, num_capsule, dim_capsule, input_num_capsule, input_dim_capsule,routings):
        inputs_expand = torch.unsqueeze(inputs, -3)
        CW = torch.unsqueeze(torch.unsqueeze(CW, 0), -1).repeat(num_capsule,1,1,input_num_capsule)
        inputs_tiled = inputs_expand.repeat([1, num_capsule,1,1])
        define_list = []
        for i in range(len(inputs_tiled)):
            define_list.append(torch.reshape((torch.matmul(CW, torch.reshape(inputs_tiled[i],(num_capsule, input_dim_capsule,input_num_capsule ,1)))),
                              (num_capsule,input_dim_capsule,dim_capsule)))
        inputs_hat = torch.stack(define_list)
        b = torch.zeros([inputs_hat.shape[0], num_capsule, dim_capsule]).to(device)
        assert routings > 0, 'The routings should be > 0.'
        for i in range(routings):
            c = self.softmax(b, dim=1)
            c = torch.reshape(c, (inputs_hat.shape[0], num_capsule, dim_capsule, 1))
            outputs = self.squash_v1(torch.reshape(torch.matmul(inputs_hat, c),(inputs_hat.shape[0],num_capsule,input_dim_capsule)))
            if i < routings - 1:
                b += torch.reshape(torch.matmul(torch.transpose(inputs_hat, 2, 3), torch.reshape(outputs, (inputs_hat.shape[0], num_capsule,input_dim_capsule,  1))),
                                  (inputs_hat.shape[0], num_capsule, dim_capsule))
        return outputs
    #############################################################################################################
