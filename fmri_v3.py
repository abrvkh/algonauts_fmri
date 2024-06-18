# import things
from einops import rearrange
import os
import numpy as np
from matplotlib import pyplot as plt
import math
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
import scipy
from PIL import Image
import gpsm
import gc
from tqdm import tqdm
from pathlib import Path
from transformers import Dinov2Config, Dinov2Model, AutoImageProcessor
PATH = '...'

def cleanup(): 
    try: 
        del fmri_model
    except NameError:
        pass
    try:
        del embed_model
    except NameError:
        pass
    try:
        del image_processor
    except NameError:
        pass
    try:
        del d
    except NameError:
        pass
    try:
        del y_l
    except NameError:
        pass
    try:
        del y_r
    except NameError:
        pass
    try:
        del features
    except NameError:
        pass
    try: 
        del out_lh
    except NameError:
        pass 
    try:
        del out_rh
    except NameError:
        pass

def select_roi_fmri(data_dir, fmri_full, hem, roi):
  # Define the ROI class based on the selected ROI
  if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
    roi_class = 'prf-visualrois'
  elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
      roi_class = 'floc-bodies'
  elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
      roi_class = 'floc-faces'
  elif roi in ["OPA", "PPA", "RSC"]:
      roi_class = 'floc-places'
  elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
      roi_class = 'floc-words'
  elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
      roi_class = 'streams'

  # Load the ROI brain surface maps
  challenge_roi_class_dir = os.path.join(data_dir, 'roi_masks',
      hem[0]+'h.'+roi_class+'_challenge_space.npy')
  roi_map_dir = os.path.join(data_dir, 'roi_masks',
      'mapping_'+roi_class+'.npy')
  challenge_roi_class = np.load(challenge_roi_class_dir)
  roi_map = np.load(roi_map_dir, allow_pickle=True).item()

  # Select the vertices corresponding to the ROI of interest
  roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
  challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
  # Get the subset of fmri data
  idx_roi = np.where(challenge_roi)[0]
  the_fmri = fmri_full[:,idx_roi]

  mapping_size = the_fmri.shape[1] # number of vertices in this roi

  return the_fmri, mapping_size, idx_roi, challenge_roi

class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform, lh_fmri, rh_fmri, device):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.lh_fmri = lh_fmri[idxs] 
        self.rh_fmri = rh_fmri[idxs]
        self.device = device
        
    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(self.device)
        patch_size = 14
        size_im = (img.shape[0], int(np.ceil(img.shape[1] / patch_size) * patch_size), int(np.ceil(img.shape[2] / patch_size) * patch_size))
        paded = torch.zeros(size_im)
        paded[:, :img.shape[1], :img.shape[2]] = img
        img = paded
        _lh_fmri = torch.tensor(self.lh_fmri[idx]).to(self.device)
        _rh_fmri = torch.tensor(self.rh_fmri[idx]).to(self.device)
        # print(img.shape) torch.Size([3, 224, 224])
        return img, _lh_fmri, _rh_fmri

class Backbone_dino(torch.nn.Module):

    def __init__(self, enc_output_layer=-1):
        super().__init__()   
        
        #self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
        self.qkv_feats = {'qkv_feats':torch.empty(0)}
        self.backbone._modules["blocks"][enc_output_layer]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_qkv) 
    def hook_fn_forward_qkv(self, module, input, output):
        self.qkv_feats['qkv_feats'] = output
    def forward(self, xs):
        xs = self.backbone.get_intermediate_layers(xs)[0]
        feats = self.qkv_feats['qkv_feats']
        # Dimensions
        nh = 12 #Number of heads
        feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
        q, k, v = feats[0], feats[1], feats[2]
        q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)
        xs = q[:,1:,:]
        return xs

class PositionEmbeddingSine(torch.nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones_like(x[:,:,0]) # torch.Size([32, 256])
        not_mask = rearrange(not_mask, 'b (h w) -> b h w', h=31) # shape [bs,h,w]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # print(pos_x.shape, pos_y.shape) torch.Size([32, 128, 2, 384]) torch.Size([32, 128, 2, 384])
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) #[bs, num_pos_feats, h, w] 
        # print(pos.shape) torch.Size([32, 768, 128, 2])
        return pos.flatten(2).permute(0,2,1) #[bs, num_patches (h*w), num_pos_feats]

class TransformerDecoderLayer(torch.nn.Module):

    def __init__(self, d_model, nhead, batch_first=True, dim_feedforward=2048, dropout=0.1, normalize_before=False):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = torch.nn.functional.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None, pos = None, query_pos = None):
                     
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.masks, src_all)
    def forward_pre(self, tgt, memory, tgt_mask  = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None, pos = None, query_pos = None): 
                    
        tgt2 = self.norm1(tgt)
        
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None, pos = None, query_pos= None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class fmriMap(torch.nn.Module):
    def __init__(self, num_queries, d_model, dim_ff, num_lh, num_rh, masks_lh, masks_rh, norm_first, nhead):
        super(fmriMap, self).__init__()
        # self.embed_qlayer = torch.nn.Linear(2, d_model)
        self.num_queries = num_queries
        self.query_embed = torch.nn.Embedding(num_queries * 2, d_model) # num_queries is #ROIs; for each hem
        # self.embed_flayer = torch.nn.Linear(768, d_model)
        self.transformer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_ff, dropout=0.1, normalize_before=norm_first)
        self.linear_lh = torch.nn.Linear(d_model, num_lh)
        self.linear_rh = torch.nn.Linear(d_model, num_rh)
        self.norm = torch.nn.LayerNorm(d_model)
        self.masks_lh = masks_lh
        self.masks_rh = masks_rh
        N_steps = 768//2
        self.pos_embed = PositionEmbeddingSine(N_steps, normalize=True) # Create position encoding model 

    def forward(self, features, queries):
        '''
        features is of shape [batch_size, num_layers, seq_len, hidden_dim]
        queries is of shape [num_queries, query_dim]
        pos is of shape [batch_size, hidden_dim, seq_len]
        '''
        # features = features.transpose(1,2).flatten(2) # to [batch_size, seq_len, hidden_dim*num_layers]
        # features = features.flatten(2).transpose(1,2) # to [batch_size, seq_len * hidden_dim, num_layers]; note that seq_len * hidden_dim plays role of seq_len and num_layers of hid_size in the transformer
        # features = features[:,:,:,0].transpose(1,2)
        # print(features.shape)
        # embed features 
        # from [batch_size, seq_len, d_model] where seq_len is nr of patches and hidden_dim is DinoV2s dimension (768)
        #features = features.transpose(1,2)
        
        # normalise 
        # features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # pass through linear embed layer
        # features_e = self.embed_flayer(features)
        # to [batch_size, seq_len, d_model]
        # print(features_e.shape)
        # unsqueeze queries
        # queries = queries.unsqueeze(0).repeat(features.shape[0], 1, 1)
        # print(queries.shape)
        # to [batch_size, num_queries, query_dim]; note num_queries plays role of seq_len (nr patches) in the transformer

        # add positional embedding 
        # print(features.shape) torch.Size([32, 256, 768])
        pos = self.pos_embed(features)
        # print(pos.shape) torch.Size([32, 256, 768])

        # embed queries
        queries_e = self.query_embed(queries)
        queries_e = queries_e.unsqueeze(0).repeat(features.shape[0], 1, 1)
        # to [batch_size, num_queries, d_model]
        
        # pass through decoder transformer; tgt is the sequence to the decoder layer, memory is the sequence from the last layer of the encoder (features in my case); they have different "seq_len"s however same d_model
        out = self.transformer(tgt = torch.zeros_like(queries_e), memory = features, pos = pos, query_pos = queries_e)
        # to [batch_size, num_queries, d_model]

        # final projection layer to fmri indices; per each hemisphere
        out_lh = self.linear_lh(out[:,:self.num_queries,:]) #[bs, num_queries, num_lh]
        out_rh = self.linear_rh(out[:,self.num_queries:,:]) #[bs, num_queries, num_rh]
        # apply masks element-wise to aggregate all ROIs; can also try removing this and comparing to fmri per vertex
        out_lh = torch.sum(out_lh * self.masks_lh.repeat(out_lh.shape[0],1,1), axis=1)
        out_rh = torch.sum(out_rh * self.masks_rh.repeat(out_rh.shape[0],1,1), axis=1)

        return out_lh, out_rh

class fmriMapSimple(torch.nn.Module):
    def __init__(self, num_lh, masks_lh, num_rh, masks_rh): 
        super(fmriMapSimple, self).__init__()
        self.linear_lh = torch.nn.Linear(768*50, num_lh)
        self.linear_rh = torch.nn.Linear(768*50, num_rh)
        self.masks_lh = masks_lh
        self.masks_rh = masks_rh
    
    def forward(self, features, queries):
        features = features[:,0:50,:] # choose one patch; [batch_size, d_model]
        features = features.flatten(1)
        # normalise 
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        out_lh = self.linear_lh(features) # [batch_size, num_v]
        out_rh = self.linear_rh(features)
        # out = out * self.masks.repeat(out.shape[0],1)
        return out_lh, out_rh

def train(fmri_model, embed_model, queries, masks_lh, masks_rh, loss_fn, device, train_imgs_dataloader, optimizer, epoch, config, max_norm):
    fmri_model.train()
    for it, (d,y_l,y_r) in tqdm(enumerate(train_imgs_dataloader), total=len(train_imgs_dataloader)):
        d, y_l, y_r = d.to(device), y_l.to(device), y_r.to(device)
        # print(d.shape) torch.Size([32, 3, 425, 425])
        optimizer.zero_grad()
        
        # pass data through DinoV2
        with torch.no_grad():
            # features = embed_model(d, output_hidden_states=True).hidden_states
            # features = embed_model(d).last_hidden_state
            features = embed_model(d)
        # select subset of layers from embed_model 
        # features = torch.cat([features[i].unsqueeze(1) for i in ss_layers], dim=1) 
        
        # pass features through model 
        out_lh, out_rh = fmri_model(features, queries) # this output is of size [batch_size, num_lh] and [batch_size, num_rh]
        
        # get masked true output
        true_masked_lh = y_l * torch.sum(masks_lh, axis=0, keepdim=True).repeat(y_l.shape[0],1) # sum over roi's
        # true_masked_lh = y_l[torch.sum(masks_lh, axis=0, keepdim=True).repeat(y_l.shape[0],1) > 0].reshape(y_l.shape[0], -1)
        true_masked_rh = y_r * torch.sum(masks_rh, axis=0, keepdim=True).repeat(y_r.shape[0],1)
        # true_masked_rh = y_r[torch.sum(masks_rh, axis=0, keepdim=True).repeat(y_r.shape[0],1) > 0].reshape(y_r.shape[0], -1)
        loss = loss_fn(out_lh, true_masked_lh) + loss_fn(out_rh, true_masked_rh)
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(fmri_model.parameters(), max_norm)
        optimizer.step()

        with torch.no_grad():
            if it % 100 == 0:
                print('Epoch: {} \tIteration: {} \tLoss: {:.6f}'.format(
                    epoch, it, loss.item()))
                correlation_lh = np.zeros(out_lh.shape[1])
                correlation_rh = np.zeros(out_rh.shape[1])
                for v in range(out_lh.shape[1]):
                    correlation_lh[v] = scipy.stats.pearsonr(out_lh[:,v].detach().cpu().numpy(),true_masked_lh[:,v].detach().cpu().numpy())[0]
                for v in range(out_rh.shape[1]):
                    correlation_rh[v] = scipy.stats.pearsonr(out_rh[:,v].detach().cpu().numpy(),true_masked_rh[:,v].detach().cpu().numpy())[0]
                print('\nCorrelation lh is '+ str(np.nanmean(correlation_lh)))
                print('\nCorrelation rh is '+ str(np.nanmean(correlation_rh)))
                # it will be nan if all constants due to stdev
            
        cleanup()

def test(fmri_model, embed_model, queries, rois, masks_lh, masks_rh, loss_fn, device, val_imgs_dataloader):
    fmri_val_pred_lh = []
    fmri_val_pred_rh = []
    fmri_val_true_lh = []
    fmri_val_true_rh = []
    fmri_model.eval()
    test_loss = []
    with torch.no_grad():
        for it, (d,y_l,y_r) in tqdm(enumerate(val_imgs_dataloader), total=len(val_imgs_dataloader)):
            d, y_l, y_r = d.to(device), y_l.to(device), y_r.to(device)
            
            # pass data through DinoV2
            with torch.no_grad():
                # features = embed_model(d, output_hidden_states=True).hidden_states
                # features = embed_model(d).last_hidden_state
                features = embed_model(d)
                
            # pass features through model 
            out_lh, out_rh = fmri_model(features, queries) # this output is of size [batch_size, num_lh] and [batch_size, num_rh]
            true_masked_lh = y_l * torch.sum(masks_lh, axis=0, keepdim=True).repeat(y_l.shape[0],1) # sum over roi's
            # true_masked_lh = y_l[torch.sum(masks_lh, axis=0, keepdim=True).repeat(y_l.shape[0],1) > 0].reshape(y_l.shape[0], -1)
            true_masked_rh = y_r * torch.sum(masks_rh, axis=0, keepdim=True).repeat(y_r.shape[0],1)
            # true_masked_rh = y_r[torch.sum(masks_rh, axis=0, keepdim=True).repeat(y_r.shape[0],1) > 0].reshape(y_r.shape[0], -1)

            # append 
            fmri_val_pred_lh += [out_lh.detach().cpu()]
            fmri_val_pred_rh += [out_rh.detach().cpu()]
            fmri_val_true_lh += [true_masked_lh.detach().cpu()]
            fmri_val_true_rh += [true_masked_rh.detach().cpu()]

        fmri_val_pred_lh = torch.vstack(fmri_val_pred_lh)   
        fmri_val_pred_rh = torch.vstack(fmri_val_pred_rh)
        fmri_val_true_lh = torch.vstack(fmri_val_true_lh)
        fmri_val_true_rh = torch.vstack(fmri_val_true_rh)
        print(fmri_val_pred_lh.shape, fmri_val_true_lh.shape, fmri_val_pred_rh.shape, fmri_val_true_rh.shape)

        # get loss 
        loss = loss_fn(fmri_val_pred_lh, fmri_val_true_lh) + loss_fn(fmri_val_pred_rh, fmri_val_true_rh)

        # Empty correlation array of shape: (LH vertices)
        fmri_val_pred_lh = fmri_val_pred_lh.numpy()
        fmri_val_pred_rh = fmri_val_pred_rh.numpy()
        fmri_val_true_lh = fmri_val_true_lh.numpy()
        fmri_val_true_rh = fmri_val_true_rh.numpy()
        correlation_lh = np.zeros(fmri_val_pred_lh.shape[1])
        correlation_rh = np.zeros(fmri_val_pred_rh.shape[1])
        # Correlate each predicted vertex with the corresponding ground truth vertex
        for v in tqdm(range(fmri_val_pred_lh.shape[1])):
            correlation_lh[v] = scipy.stats.pearsonr(fmri_val_pred_lh[:,v], fmri_val_true_lh[:,v])[0]
        for v in tqdm(range(fmri_val_pred_rh.shape[1])):
            correlation_rh[v] = scipy.stats.pearsonr(fmri_val_pred_rh[:,v], fmri_val_true_rh[:,v])[0]
    return fmri_val_pred_lh, fmri_val_pred_rh, correlation_lh, correlation_rh, loss.item()

if __name__=='__main__': 
    subj = 'subj01'
    device = 'cuda'

    # LOAD DATA 
    # load fmri
    data_dir = '...' + subj
    fmri_dir = os.path.join(data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    print('LH training fMRI data shape:')
    print(lh_fmri.shape)
    print('(Training stimulus images × LH vertices)')

    print('\nRH training fMRI data shape:')
    print(rh_fmri.shape)
    print('(Training stimulus images × RH vertices)')

    # load images 
    train_img_dir  = os.path.join(data_dir, 'training_split', 'training_images')
    test_img_dir  = os.path.join(data_dir, 'test_split', 'test_images')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()
    print('Training images: ' + str(len(train_img_list)))
    print('Test images: ' + str(len(test_img_list)))

    train_img_file = train_img_list[0]
    print('Training image file name: ' + train_img_file)
    print('73k NSD images ID: ' + train_img_file[-9:-4])

    # create train, validation and test partitions
    # rand_seed = 5 
    # np.random.seed(rand_seed)

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) / 100 * 90))
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    # Assign 90% of the shuffled stimulus images to the training partition,
    # and 10% to the test partition
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # No need to shuffle or split the test stimulus images
    idxs_test = np.arange(len(test_img_list))

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('Validation stimulus images: ' + format(len(idxs_val)))
    print('Test stimulus images: ' + format(len(idxs_test)))

    batch_size = 32

    # define image transform
    transform = transforms.Compose([
        transforms.ToTensor(), 
        # transforms.Resize(244), 
        # transforms.CenterCrop(224), 
        # transforms.Normalize([0.5], [0.5])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    #transform = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform, lh_fmri, rh_fmri, device), 
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform, lh_fmri, rh_fmri, device), 
        batch_size=batch_size
    )

    # data, target_l, target_r = next(iter(train_imgs_dataloader))
    # with torch.no_grad():
    #     outputs = model(data, output_hidden_states=True)
    # hidden_states_all = outputs.hidden_states # (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer)
    # print(len(hidden_states_all), hidden_states_all[0].shape)

    #rois = ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal", "unknown"]
    rois = ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]
    #rois = ["early"]
    # get masks
    masks_lh = []
    masks_rh = []
    for roi in rois: 
        if roi != 'unknown':
            the_fmri, mapping_size, idx_roi_lh, challenge_roi_lh = select_roi_fmri(data_dir, lh_fmri, 'lh', roi)
            masks_lh += [challenge_roi_lh[np.newaxis]]
            the_fmri, mapping_size, idx_roi_rh, challenge_roi_rh = select_roi_fmri(data_dir, rh_fmri, 'rh', roi)
            masks_rh += [challenge_roi_rh[np.newaxis]]
    # add unknown vertices 
    if 'unknown' in rois: 
        unknown_lh = np.ones(lh_fmri.shape[1]) - np.sum(np.vstack(masks_lh), axis=0)
        unknown_rh = np.ones(rh_fmri.shape[1]) - np.sum(np.vstack(masks_rh), axis=0)
        masks_lh += [unknown_lh[np.newaxis]]
        masks_rh += [unknown_rh[np.newaxis]]
    # make tensor
    masks_lh = torch.tensor(np.vstack(masks_lh)).to(device) #[num rois, num vertices lh]
    masks_rh = torch.tensor(np.vstack(masks_rh)).to(device) #[num rois, num vertices rh]

    # Create embed model 
    embed_model = Backbone_dino()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    embed_model.to(device)
    embed_model.eval()

    # Create fmri model
    d_model = 768 
    nhead = 16
    norm_first = False
    dim_ff = 2048
    num_queries = len(rois)
    num_lh = lh_fmri.shape[1] #lh_fmri.shape[1], idx_roi_lh.shape[0]
    num_rh = rh_fmri.shape[1] #rh_fmri.shape[1], idx_roi_rh.shape[0]
    fmri_model = fmriMap(num_queries, d_model, dim_ff, num_lh, num_rh, masks_lh, masks_rh, norm_first, nhead).to(device)
    #fmri_model = fmriMapSimple(num_lh, masks_lh, num_rh, masks_rh).to(device)

    lr = 1e-4
    max_norm = 0.1
    loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(fmri_model.parameters(), lr=lr)
    param_dicts = [ { "params" : [ p for n , p in fmri_model.named_parameters() if p.requires_grad]}, ] 
    optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                              weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200)

    # queries = torch.nn.functional.one_hot(torch.arange(len(rois)*2)).float().to(device) # [14,14] now
    queries = torch.tensor([i for i in range(len(rois)*2)]).to(device)

    config = {'lr': lr, 'd_model': d_model, 'nhead': nhead, 'num_queries': num_queries, 'batch_size': batch_size, 'rois': rois, 'epochs': 100, 'dim_ff': dim_ff, 'norm_first':norm_first}
    
    #'''
    with open('./fmri/val_results.txt', 'w') as f:
        f.write(f'config:' + str(config) + '\n')
        f.write(f'validation results: \n') 
    
    for epoch in range(0, 10):
        train(fmri_model, embed_model, queries, masks_lh, masks_rh, loss_fn, device, train_imgs_dataloader, optimizer, epoch, config, max_norm)
        torch.save(fmri_model, f"fmri/ckpt/ana_{epoch}.pt")
        # run pass over validation data
        fmri_val_pred_lh, fmri_val_pred_rh, correlation_lh, correlation_rh, loss = test(fmri_model, embed_model, queries, rois, masks_lh, masks_rh, loss_fn, device, val_imgs_dataloader)
        # compute mean ignoring nan
        print('LH correlation mean: ' + str(np.nanmean(correlation_lh)))
        print('RH correlation mean: ' + str(np.nanmean(correlation_rh)))
        print('Loss: ' + str(loss))
        # learning rate change 
        lr_scheduler.step()
        # write to file to store results 
        with open('./fmri/val_results.txt', 'a') as f:
            f.write(f'epoch {epoch}, val_mean_corr_lh: {np.nanmean(correlation_lh)}, val_mean_corr_rh: {np.nanmean(correlation_rh)}, loss: {loss} \n') 
    #'''
    '''
    # load a model 
    fmri_model = torch.load("fmri/ckpt/ana_1.pt")
    fmri_val_pred_lh, fmri_val_pred_rh, correlation_lh, correlation_rh, loss = test(fmri_model, embed_model, queries, rois, masks_lh, masks_rh, loss_fn, device, val_imgs_dataloader)
    # compute mean ignoring nan
    print('LH correlation mean: ' + str(np.nanmean(correlation_lh)))
    print('RH correlation mean: ' + str(np.nanmean(correlation_rh)))
    print('Loss: '+str(loss))
    '''
