import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_normal_

from layer import MLPLayers, SequenceAttLayer
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
model_path = "bert-small"
BERT_DIM= 512
SEM_DIM = 64
NUM_INCENTIVE =3
PAD = 0
EMBEDDING_SIZE = 30000000
WORD_EMBEDDING_SIZE= 1200000


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.phase = ""

        # get field names and parameter value from config
        self.embedding_size = config.embedding_size
        self.max_his_len = config.max_his_len
        self.l2_reg = config.l2_reg
        self.loss = nn.BCELoss()

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='', help='Model save path.')
        parser.add_argument('--buffer', type=int, default=1, help='Whether to buffer feed dicts for dev/test')
        return parser

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input):
        out_dict = {}
        out_dict["scores"] = 0
        out_dict["summary_dic"] = {}
        out_dict["embed_hook"] = {}
        return out_dict

    def predict(self, input):
        out_dict = self.forward(input)
        scores = out_dict["scores"]
        embed = out_dict["embed_hook"]
        return scores, embed

    def calculate_loss(self, input):
        out_dict = self.forward(input)
        summary_dic = out_dict["summary_dic"]
        scores = out_dict["scores"]
        loss = self.loss(scores.squeeze(-1), input.label)
        if out_dict["types_sum"] is not None:
            types_sum = out_dict["types_sum"]
            loss += types_sum.float().var() / (types_sum.float().mean() ** 2 + 1e-10)
        loss = loss + self.l2_reg * (self.embedding_layer.weight ** 2).sum()
        return loss, summary_dic

    def set_phase(self, phase):
        self.phase = phase

    def lalign(self, x, y, alpha=2):
        # bsz : batch size (number of positive pairs)
        # d : latent dim # x : Tensor, shape=[bsz, d]
        # latents for one side of positive pairs
        # y : Tensor, shape=[bsz, d]
        # latents for the other side of positive pairs
        # lam : hyperparameter balancing the two losses
        return (x - y).norm(dim=1).pow(alpha).mean()

    def lunif(self, x, t=2):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def inverse_gumbel_cdf(self, y, mu, beta):
        return mu - beta * np.log(-np.log(y))

    def gumbel_softmax_sampling(self, h, mu=0, beta=1, tau=1e-1):
        """
        h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
        """
        shape_h = h.shape
        p = F.softmax(h, dim=1)
        y = torch.rand(shape_h) + 1e-25  # ensure all y is positive.
        g = self.inverse_gumbel_cdf(y, mu, beta)
        x = torch.log(p) + g.cuda()  # samples follow Gumbel distribution.
        # using softmax to generate one_hot vector:
        x = x / tau
        x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
        return x


class PRINT(BaseModel):
    def __init__(self, config):
        super(PRINT, self).__init__(config)

        self.embedding_layer = nn.Embedding(EMBEDDING_SIZE, self.embedding_size, padding_idx=PAD)
        self.word_embedding = nn.Embedding(WORD_EMBEDDING_SIZE, BERT_DIM, padding_idx=PAD)
        self.seq_att_layer = SequenceAttLayer(att_hidden_size=(4 * self.embedding_size, 64, 16), activation='dice',softmax_stag=True, return_seq_weight=False)

        self.dnn_mlp_layers = MLPLayers([30 * self.embedding_size + SEM_DIM, 128, 64], activation='relu', dropout=0.1, bn=False)
        self.dnn_predict_layers = nn.Linear(64, 1)
        self.out = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.bert = BertModel.from_pretrained(model_path)
        self.bert.set_input_embeddings(self.word_embedding)
        self.bert_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(BERT_DIM, SEM_DIM),
            nn.ReLU(),
        )

        self.Incentive_Discerning_Network = nn.Sequential(
            nn.BatchNorm1d(4*SEM_DIM),
            MLPLayers([4*SEM_DIM, 128, 64], activation='relu', dropout=0, bn=False),
            nn.Linear(64, NUM_INCENTIVE)
        )
        self.Incentive_Intensity_Network_0 = nn.Sequential(
            nn.BatchNorm1d(4*SEM_DIM),
            MLPLayers([4*SEM_DIM, 128, 64], activation='relu', dropout=0, bn=False),
            nn.Linear(64, 1)
            )
        self.Incentive_Intensity_Network_1 =nn.Sequential(
            nn.BatchNorm1d(4 * SEM_DIM),
            MLPLayers([4*SEM_DIM, 128, 64], activation='relu', dropout=0, bn=False),
            nn.Linear(64, 1)
            )
        self.Incentive_Intensity_Network_2 = None
        # parameters initialization
        self.apply(self._init_weights)
        self.softplus =nn.Softplus()
        self.w_noise = nn.Parameter(torch.zeros(4 * SEM_DIM, NUM_INCENTIVE), requires_grad=True)
        for name, parameter in  self.bert.encoder.named_parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input):
        B,L = input.AdIDList.size()

        # target
        target_emb = self.embedding_layer(input.AdID%20000000)  # [B,E]

        # AdvertiserID
        advertiser_id_emb = self.embedding_layer(input.AdvertiserID%20000000 + 2 ** 20)  # [B,E]

        # Depth
        depth_emb = self.embedding_layer(input.Depth)  # [B,E]

        # Position
        position_emb = self.embedding_layer(input.Position)  # [B,E]

        # DescriptionID
        description_id_emb = self.embedding_layer(input.DescriptionID%20000000 + 2 ** 20 * 2)  # [B,E]

        # user
        user_emb = self.embedding_layer(input.user_id%20000000 + 2 ** 20 * 3)  # [B,E]

        # query
        query_emb = self.embedding_layer(input.QueryID%20000000 + 2 ** 20 * 4)  # [B,E]

        # keyword
        keyword_emb = self.embedding_layer(input.KeywordID%20000000 + 2 ** 20 * 5)  # [B,E]

        # title
        title_emb = self.embedding_layer(input.TitleID%20000000 + 2 ** 20 * 6)  # [B,E]

        # TitleToken
        title_token_emb = self.embedding_layer(input.TitleToken%20000000 + 2 ** 20 * 7).view(B, -1)  # [B,10E]

        # QueryToken
        query_token_emb = self.embedding_layer(input.QueryToken%20000000 + 2 ** 20 * 8).view(B, -1)  # [B,10E]


        # seq
        adidlist_feat = input.AdIDList.view(B,-1)
        adidlist_emb = self.embedding_layer(adidlist_feat)  # [B,L,E]


        # concat all query-ad pair
        queryTokenList = torch.cat([input.QueryToken,input.QueryTokenList.view(B*L, -1)],dim=0)
        titleTokenList = torch.cat([input.TitleToken,input.TitleTokenList.view(B*L, -1)],dim=0)


        input_ids,input_mask,token_type_ids = self.tokenization(queryTokenList,titleTokenList, B*(L+1))
        bert_sequence_out = self.bert(input_ids=input_ids,attention_mask=input_mask,token_type_ids=token_type_ids)[0]


        ad_hoc_bert_sequence_out,history_bert_sequence_out = torch.split(bert_sequence_out, [B,B*L], dim=0)

        ad_hoc_relevance = ad_hoc_bert_sequence_out[:, 0].view(B,-1)
        history_relevance = history_bert_sequence_out[:, 0].view(B,L,-1)
        query_semantic_embedding = torch.mean(ad_hoc_bert_sequence_out[:,1:1+L],dim=1).view(B,-1)
        history_query_semantic_embedding = torch.mean(history_bert_sequence_out[:,1:1+L],dim=1).view(B,L,-1)

        intent_aware_relevance_preference = self.attention(query_semantic_embedding.view(B,1,-1), history_query_semantic_embedding, history_relevance).view(B,-1)
        category_aware_relevance_Preference = self.attention(target_emb.view(B,1,-1), adidlist_emb, history_relevance).view(B,-1)

        relevance_Preference = torch.mul(intent_aware_relevance_preference, category_aware_relevance_Preference)

        relevance_Preference = self.bert_layer(relevance_Preference)
        ad_hoc_relevance = self.bert_layer(ad_hoc_relevance)

        relevance_inputs = torch.cat([ad_hoc_relevance,relevance_Preference,ad_hoc_relevance-relevance_Preference,torch.mul(ad_hoc_relevance, relevance_Preference)], dim=1).view(B,-1)
        logits, top_types = self.noisy_logits( self.Incentive_Discerning_Network(relevance_inputs),self.training)
        t = self.gumbel_softmax_sampling(logits)

        y_0 = self.softplus(self.Incentive_Intensity_Network_0(relevance_inputs))
        y_1 = -self.softplus(self.Incentive_Intensity_Network_1(relevance_inputs))
        y_2 = torch.zeros([B,1]).cuda()
        y = torch.cat([y_0,y_1,y_2],dim=1)
        z = torch.sum(torch.mul(t, y), dim=1,keepdim=True)

        # target attention
        mask = (adidlist_feat == 0)  # [B, L]
        pool_emb_adidlist = self.seq_att_layer(target_emb, adidlist_emb, mask)

        # concatenate

        mlp_input = torch.cat(
            (user_emb, target_emb, advertiser_id_emb, depth_emb, position_emb, description_id_emb, query_emb,
             keyword_emb, title_emb, title_token_emb, query_token_emb, pool_emb_adidlist, ad_hoc_relevance), dim=1
        )  # 30E

        # predict_layer
        output = self.dnn_mlp_layers(mlp_input)
        output = self.dnn_predict_layers(output)

        output += z
        print(t[0],z[0],output[0])
        output = self.out(output)

        out_dict = {}
        out_dict["summary_dic"] = {}
        out_dict["embed_hook"] = {
            'mlp_input':mlp_input
        }
        out_dict["scores"] = output
        out_dict["types_sum"] = top_types.sum(0)
        return out_dict

    def tokenization(self, queryTokenList, titleTokenList, L):
        cls = torch.ones([L,1],dtype=torch.int) * 101
        sep1 = torch.ones([L,1],dtype=torch.int) * 102
        sep2 = torch.ones([L,1],dtype=torch.int) * 102
        padding = torch.zeros([L, 32-3-2*10],dtype=torch.int)
        cls,sep1,sep2,padding,queryTokenList, titleTokenList  = cls.cuda(),sep1.cuda(), sep2.cuda(), padding.cuda(), queryTokenList.cuda(), titleTokenList.cuda()
        input_ids = torch.cat([cls, queryTokenList, sep1, titleTokenList, sep2, padding], dim=1).int().cuda()
        input_mask =(~torch.eq(input_ids,0)).int().cuda()
        token_type_ids = (torch.cat([torch.zeros([L,12]),torch.ones([L,11]),torch.zeros([L,9])], dim=1)).int().cuda()

        return input_ids, input_mask, token_type_ids


    def attention(self, query, key, value):
        similarity =  torch.cosine_similarity(query, key, dim=-1)
        weight =  torch.softmax(similarity, dim=1)
        return torch.mean(torch.matmul(weight, value),dim=1)

    def noisy_logits(self, clean_logits, train, noise_ratio=0.02):
        top_types = torch.softmax(clean_logits,dim=1)
        if  train:
            logits = torch.randn_like(clean_logits) * noise_ratio + clean_logits
        else:
            logits = clean_logits
        return logits, top_types





