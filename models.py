import torch
import math
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils_eff import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    Swish,
    MemoryEfficientSwish,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

def H_cv2torch_batch_sim(temp_Hm, w, h):
    b,_,_ = temp_Hm.shape
    t_np = np.array(
            [[[2 / w, 0, -1],
              [0, 2 / h, -1],
              [0, 0, 1]]]*b
              )

    T = torch.from_numpy(t_np).float().to(device)
    theta = torch.linalg.inv(T@temp_Hm@torch.linalg.inv(T))
    return theta[:,:2,:]

class EfficientNet_cycle_ransac(nn.Module):
    def __init__(self, blocks_args=None, global_params=None, pad_size=200, len_diff=1024):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.pad_size = pad_size

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        # self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._conv_stem = nn.Sequential(
                                        Conv2d(3, out_channels, kernel_size=(3,3), stride=2, bias=False),
                                        nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
                                        MemoryEfficientSwish(),
                                        Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, bias=False),
                                        nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
                                        MemoryEfficientSwish(),
                                        Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, bias=False),
                                        nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
                                        MemoryEfficientSwish(),
                                        )

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # feature Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._swish = MemoryEfficientSwish()


        ## predict head
        self.att_mask_sim = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=4, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),

            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=8, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=16, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),


            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.feature_diff = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),

            
            nn.Conv2d(out_channels, len_diff*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=len_diff*2, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),
            nn.Conv2d(len_diff*2, len_diff*2, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(num_features=len_diff*2, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),

            nn.AdaptiveAvgPool2d(output_size=1),
        )

        self.diff_estimator = nn.Sequential(
            nn.Conv2d(len_diff*2, len_diff*2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=len_diff*2, momentum=bn_mom, eps=bn_eps),
            MemoryEfficientSwish(),
            nn.Conv2d(len_diff*2, len_diff*2, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=1),
            nn.Tanh(),
        )

        self.diff_score_estimator = nn.Sequential(
            nn.Linear(len_diff*4, 512),
            MemoryEfficientSwish(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self.diff2homography_grid = nn.Sequential(
            nn.Linear(len_diff*4, 512),
            MemoryEfficientSwish(),
            nn.Linear(512,6),
            nn.Tanh(),
        )


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x_att_mask = self.att_mask_sim(inputs)
        # Stem
        input_shape = inputs.shape
        x_att_mask = F.interpolate(x_att_mask, size=input_shape[2:], mode='nearest')
        x = self._conv_stem(inputs*(x_att_mask+1))
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        return x, x_att_mask
    
    def extract_visual(self, inputs):
        list_visuals = []
        x_att_mask = self.att_mask_sim(inputs)
        # Stem
        input_shape = inputs.shape
        x_att_mask = F.interpolate(x_att_mask, size=input_shape[2:], mode='nearest')
        x = self._conv_stem(inputs*(x_att_mask+1))
        list_visuals.append(x)
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            list_visuals.append(x)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        list_visuals.append(x)
        return x, x_att_mask, list_visuals

    def estimate_H_sim(self, src, dst, num_combine_ids):
        M = torch.zeros((num_combine_ids, 3,3)).to(device)
        x1s,y1s = src[:,0,0],src[:,0,1]
        x2s,y2s = src[:,1,0],src[:,1,1]
        X1s,Y1s = dst[:,0,0],dst[:,0,1]
        X2s,Y2s = dst[:,1,0],dst[:,1,1]

        smooth = 1e-16
        dxs = x1s-x2s
        dys = y1s-y2s
        dXs = X1s-X2s
        dYs = Y1s-Y2s
        x1y2s = x1s*y2s
        x2y1s = x2s*y1s

        d = 1.0/(dxs*dxs+smooth+dys*dys+smooth)
        S0 = d*(dXs*dxs+dYs*dys)
        S1 = d*(dYs*dxs-dXs*dys)
        S2 = d*( dYs*(x1y2s-x2y1s)-(X1s*y2s-X2s*y1s)*dys - (X1s*x2s-X2s*x1s)*dxs)
        S3 = d*(-dXs*(x1y2s-x2y1s)-(Y1s*x2s-Y2s*x1s)*dxs - (Y1s*y2s-Y2s*y1s)*dys)

        ## S0  -S1  S2 
        ## S1  S0   S3
        ## 0   0    1
        M[:,0,0] = S0
        M[:,0,1] = -S1
        M[:,0,2] = S2

        M[:,1,0] = S1
        M[:,1,1] = S0
        M[:,1,2] = S3

        M[:,2,2] += 1

        return M


    def compute_err_sim(self, src, dst, H):
        src_ex = torch.cat([src, torch.ones_like(src[:,:1])], axis=-1).float()
        H_t = H[:,:2,:].permute(0,2,1)
        src_dst = torch.matmul(src_ex, H_t)
        err=torch.linalg.norm(src_dst-dst, axis=-1)
        return err
    
    def ransac(self, pts_t, pts_p):
        pts_t = pts_t.view(-1,2)
        pts_p = pts_p.view(-1,2)
        max_trials = 100
        err_threshold = math.sqrt(5.0)


        best_inliers = []

        num_samples = pts_t.shape[0]
        spl_idxs = torch.multinomial(torch.ones(num_samples), max_trials, replacement=False)

        combine_ids = torch.combinations(spl_idxs, 2)
        data_1_combine = pts_t[combine_ids]
        data_2_combine = pts_p[combine_ids]
        num_combine_ids = len(combine_ids)

        H_combine = self.estimate_H_sim(data_1_combine, data_2_combine, num_combine_ids)
        err_combine = self.compute_err_sim(pts_t, pts_p, H_combine)

        inliers_combine = err_combine < err_threshold
        inliers_count_combine = torch.count_nonzero(inliers_combine, dim=-1)
        errs_sum_combine = torch.sum(err_combine, dim=-1)

        best_id = torch.argmax(inliers_count_combine+(1-errs_sum_combine/torch.max(errs_sum_combine)), dim=-1)

        best_H = H_combine[best_id]
        best_inliers = inliers_combine[best_id]
        return best_H, best_inliers

    def estimate_H_batch(self, src, dst, num_combine_ids, batch):
        M = torch.zeros((batch, num_combine_ids, 3, 3)).to(device)
        x1sb, y1sb = src[:,:,0,0], src[:,:,0,1]
        x2sb, y2sb = src[:,:,1,0], src[:,:,1,1]
        X1sb, Y1sb = dst[:,:,0,0], dst[:,:,0,1]
        X2sb, Y2sb = dst[:,:,1,0], dst[:,:,1,1]

        smooth = 1e-24
        dxsb = x1sb-x2sb
        dysb = y1sb-y2sb
        dXsb = X1sb-X2sb
        dYsb = Y1sb-Y2sb
        x1y2sb = x1sb*y2sb
        x2y1sb = x2sb*y1sb
        
        d = 1.0/(dxsb*dxsb+smooth+dysb*dysb+smooth)
        S0 = d*(dXsb*dxsb+dYsb*dysb)
        S1 = d*(dYsb*dxsb-dXsb*dysb)
        S2 = d*( dYsb*(x1y2sb-x2y1sb)-(X1sb*y2sb-X2sb*y1sb)*dysb-(X1sb*x2sb-X2sb*x1sb)*dxsb)
        S3 = d*(-dXsb*(x1y2sb-x2y1sb)-(Y1sb*x2sb-Y2sb*x1sb)*dxsb-(Y1sb*y2sb-Y2sb*y1sb)*dysb)

        M[:,:,0,0] = S0
        M[:,:,0,1] = -S1
        M[:,:,0,2] = S2

        M[:,:,1,0] = S1
        M[:,:,1,1] = S0
        M[:,:,1,2] = S3

        M[:,:,2,2] += 1

        return M

    def compute_err_batch(self, src, dst, H):
        src_ex_batch = torch.cat([src, torch.ones_like(src[:,:,:1])], dim=-1).float()
        H_t_batch = H[:,:,:2,:].permute(0,1,3,2)
        # list_src_dst = []
        list_errs = []
        for src_ex, H_t, temp_dst in zip(src_ex_batch, H_t_batch, dst):
            temp_src_dst = torch.matmul(src_ex, H_t)
            err = torch.linalg.norm(temp_src_dst-temp_dst, axis=-1)
            list_errs.append(err)
        errs = torch.stack(list_errs, dim=0)
        return errs

    def ransac_batch_sim(self, pts_t_batch, pts_p_batch):
        batch, num_pts = pts_t_batch.shape[:2]
        pts_t_batch = pts_t_batch.view(batch,num_pts,2)
        pts_p_batch = pts_p_batch.view(batch,num_pts,2)
        max_trials = 128
        err_threshold = math.sqrt(5.0)

        num_samples = num_pts
        spl_idxs = torch.multinomial(torch.ones(num_samples), max_trials, replacement=False)

        combine_ids = torch.combinations(spl_idxs, 2)
        data_1_combine = pts_t_batch[:,combine_ids]
        data_2_combine = pts_p_batch[:,combine_ids]
        num_combine_ids = len(combine_ids)

        H_combine = self.estimate_H_batch(data_1_combine, data_2_combine, num_combine_ids, batch)
        err_combine = self.compute_err_batch(pts_t_batch, pts_p_batch, H_combine)

        inliers_combine = err_combine < err_threshold
        inliers_count_combine = torch.count_nonzero(inliers_combine, dim=-1)
        errs_sum_combine = torch.sum(err_combine, dim=-1)

        best_ids = torch.argmax(inliers_count_combine+(1-errs_sum_combine/torch.max(errs_sum_combine)), dim=-1)

        list_best_H = []
        # list_best_inliers = []
        for best_id, temp_H_combine, temp_inliers in zip(best_ids, H_combine, inliers_combine):
            list_best_H.append(temp_H_combine[best_id])
            # list_best_inliers.append(temp_inliers[best_id])
        
        best_H_batch = torch.stack(list_best_H, dim=0)
        # best_inliers_batch = torch.stack(list_best_inliers, dim=0)
        

        
        return best_H_batch

    def ransac_batch_sim_train(self, pts_t_batch, pts_p_batch):
        batch, num_pts = pts_t_batch.shape[:2]
        pts_t_batch = pts_t_batch.view(batch,num_pts,2)
        pts_p_batch = pts_p_batch.view(batch,num_pts,2)
        max_trials = 256


        num_samples = num_pts
        spl_idxs = torch.multinomial(torch.ones(num_samples), max_trials, replacement=False)

        combine_ids = torch.combinations(spl_idxs, 2)
        data_1_combine = pts_t_batch[:,combine_ids]
        data_2_combine = pts_p_batch[:,combine_ids]
        num_combine_ids = len(combine_ids)

        H_combine = self.estimate_H_batch(data_1_combine, data_2_combine, num_combine_ids, batch)
        
        best_H_batch = torch.mean(H_combine, dim=1)
        # print(best_H_batch.shape)
        # best_H_batch = torch.squeeze(best_H_batch)
        # print(best_H_batch.shape)

        
        return best_H_batch

    def ransac_batch(self, pts_t_batch, pts_p_batch):
        list_pred_hs = []
        for pts_t, pts_p in zip(pts_t_batch, pts_p_batch):
            temp_h,_ = self.ransac(pts_t, pts_p)
            list_pred_hs.append(temp_h)
        H_batch = torch.stack(list_pred_hs, dim=0)
        return H_batch

    def forward_old(self, xt, xa, m_a, m_t, pts_t):
        b,c,h,w = xt.shape

        if random.random() < 0.3:
            temp_scale = random.uniform(0.5, 0.8)
            xt_re = F.interpolate(xt, scale_factor=temp_scale, mode='bilinear', align_corners=True)
            xa_re = F.interpolate(xa, scale_factor=temp_scale, mode='bilinear', align_corners=True)
        else:
            xt_re = xt.clone()
            xa_re = xa.clone()

        ft, _ = self.extract_features(xt_re)
        fa, _ = self.extract_features(xa_re)

        pad_size = self.pad_size
        # homography 
        f_ta = torch.cat([ft, fa], dim=1)
        f_at = torch.cat([fa, ft], dim=1)
        # get the grid points diff 
        features_diff = self.feature_diff(f_ta)
        pred_diff = self.diff_estimator(features_diff)
        features_diff_inv = self.feature_diff(f_at)
        pred_diff_inv  = self.diff_estimator(features_diff_inv)

        # generate noise predict for score
        # shape_diff = pred_diff.shape
        # if random.random() < 0.5:
        #     pred_diff_fake = pred_diff + 0.01*torch.randn(shape_diff).to(device)
        # else:
        #     pred_diff_fake = pred_diff - 0.01*torch.randn(shape_diff).to(device)
        # features_diff = features_diff.view(b, -1)
        # pred_diff_score = self.diff_score_estimator(torch.cat([features_diff, pred_diff], dim=1))
        # pred_diff_fake_score = self.diff_score_estimator(torch.cat([features_diff, pred_diff_fake], dim=1))

        # get the homography matrix(no pad) from pd
        pts_t = pts_t.view(b,-1,2)
        pts_diff = pred_diff.view(b,-1,2)
        pts_diff_scale = pts_diff.clone()
        pts_diff_scale[:,:,:1] = pts_diff[:,:,:1]*640
        pts_diff_scale[:,:,1:] = pts_diff[:,:,1:]*480

        pts_p = pts_t + pts_diff_scale
        H_nopad = self.ransac_batch_sim_train(pts_p, pts_t)
        H_theta_nopad = H_cv2torch_batch_sim(H_nopad, w, h)

        H_nopad_inv = self.ransac_batch_sim_train(pts_t, pts_p)
        H_theta_nopad_inv = H_cv2torch_batch_sim(H_nopad_inv, w, h)
        


        # get the homography matirx(with pad) from pd
        H_pad = self.ransac_batch_sim_train(pts_p+pad_size, pts_t+pad_size)
        H_theta_pad = H_cv2torch_batch_sim(H_pad, w+2*pad_size, h+2*pad_size)

        
        H_pad_inv = self.ransac_batch_sim_train(pts_t+pad_size, pts_p+pad_size)
        H_theta_pad_inv = H_cv2torch_batch_sim(H_pad_inv, w+2*pad_size, h+2*pad_size)
        
        
        # affine preditct mask_t, mask_a
        grid = F.affine_grid(H_theta_pad[:,:2,:], m_a.shape, align_corners=True)
        pred_m_a = torch.clamp(F.grid_sample(m_a, grid, align_corners=True), 0.0, 1.0)

        grid = F.affine_grid(H_theta_pad_inv[:,:2,:], m_a.shape, align_corners=True)
        pred_m_t = torch.clamp(F.grid_sample(m_t, grid, align_corners=True), 0.0, 1.0)

    
        # return pred_diff, pred_diff_inv, \
        #         pred_diff_fake_score, pred_diff_score, \
        #         H_theta_pad[:,:2,:], H_theta_nopad[:,:2,:], H_theta_pad_inv[:,:2,:], H_theta_nopad_inv[:,:2,:], \
        #         pred_m_a,  pred_m_t
        
        return pred_diff, pred_diff_inv, \
                H_theta_pad[:,:2,:], H_theta_nopad[:,:2,:], H_theta_pad_inv[:,:2,:], H_theta_nopad_inv[:,:2,:], \
                pred_m_a,  pred_m_t

    def forward_cycle(self, xt, xa, m_a, m_t, pts_t, train_flag='w'):
        b,c,h,w = xt.shape

        if random.random() < 0.3:
            temp_scale = random.uniform(0.5, 0.8)
            xt_re = F.interpolate(xt, scale_factor=temp_scale, mode='bilinear', align_corners=True)
            xa_re = F.interpolate(xa, scale_factor=temp_scale, mode='bilinear', align_corners=True)
        else:
            xt_re = xt.clone()
            xa_re = xa.clone()

        ft, _ = self.extract_features(xt_re)
        fa, _ = self.extract_features(xa_re)

        pad_size = self.pad_size
        # homography 
        f_ta = torch.cat([ft, fa], dim=1)
        f_at = torch.cat([fa, ft], dim=1)
        # get the grid points diff 
        features_diff = self.feature_diff(f_ta)
        pred_diff = self.diff_estimator(features_diff)
        features_diff_inv = self.feature_diff(f_at)
        pred_diff_inv  = self.diff_estimator(features_diff_inv)

        # generate noise predict for score
        # shape_diff = pred_diff.shape
        # if random.random() < 0.5:
        #     pred_diff_fake = pred_diff + 0.01*torch.randn(shape_diff).to(device)
        # else:
        #     pred_diff_fake = pred_diff - 0.01*torch.randn(shape_diff).to(device)
        # features_diff = features_diff.view(b, -1)
        # pred_diff_score = self.diff_score_estimator(torch.cat([features_diff, pred_diff], dim=1))
        # pred_diff_fake_score = self.diff_score_estimator(torch.cat([features_diff, pred_diff_fake], dim=1))

        # get the homography matrix(no pad) from pd
        if train_flag == 'l':
            pts_t = pts_t.view(b,-1,2)
            pts_diff = pred_diff.view(b,-1,2)
            pts_diff_scale = pts_diff.clone()
            pts_diff_scale[:,:,:1] = pts_diff[:,:,:1]*640
            pts_diff_scale[:,:,1:] = pts_diff[:,:,1:]*480

            pts_p = pts_t + pts_diff_scale
            H_nopad = self.ransac_batch_sim(pts_p, pts_t)
            H_theta_nopad_l = H_cv2torch_batch_sim(H_nopad, w, h)

            H_nopad_inv = self.ransac_batch_sim(pts_t, pts_p)
            H_theta_nopad_inv_l = H_cv2torch_batch_sim(H_nopad_inv, w, h)
        else:
            temp_pad_diff = torch.zeros_like(pred_diff)
            H_theta_nopad_w = self.diff2homography_grid(torch.cat([pred_diff, temp_pad_diff], dim=1))
            H_theta_nopad_w = H_theta_nopad_w.view(-1,2,3)

            H_theta_nopad_inv_w = self.diff2homography_grid(torch.cat([pred_diff_inv, temp_pad_diff], dim=1))
            H_theta_nopad_inv_w = H_theta_nopad_inv_w.view(-1,2,3)
        
        if train_flag == 'w':
            H_theta_nopad = H_theta_nopad_w
            H_theta_nopad_inv = H_theta_nopad_inv_w
        elif train_flag == 'l':
            H_theta_nopad = H_theta_nopad_l
            H_theta_nopad_inv = H_theta_nopad_inv_l
        elif train_flag == 'w_l':
            H_theta_nopad = H_theta_nopad_l*0.1+H_theta_nopad_w*0.9
            H_theta_nopad_inv = H_theta_nopad_inv_l*0.1+H_theta_nopad_inv_w*0.9

        


        # get the homography matirx(with pad) from pd
        if train_flag == 'l':
            H_pad = self.ransac_batch_sim(pts_p+pad_size, pts_t+pad_size)
            H_theta_pad_l = H_cv2torch_batch_sim(H_pad, w+2*pad_size, h+2*pad_size)

            H_pad_inv = self.ransac_batch_sim(pts_t+pad_size, pts_p+pad_size)
            H_theta_pad_inv_l = H_cv2torch_batch_sim(H_pad_inv, w+2*pad_size, h+2*pad_size)
        else:
            temp_pad_diff = torch.ones_like(pred_diff)
            H_theta_pad_w = self.diff2homography_grid(torch.cat([pred_diff, temp_pad_diff], dim=1))
            H_theta_pad_w = H_theta_pad_w.view(-1,2,3)

            H_theta_pad_inv_w = self.diff2homography_grid(torch.cat([pred_diff_inv, temp_pad_diff], dim=1))
            H_theta_pad_inv_w = H_theta_pad_inv_w.view(-1,2,3)

        
        if train_flag == 'w':
            H_theta_pad = H_theta_pad_w
            H_theta_pad_inv = H_theta_pad_inv_w
        elif train_flag == 'l':
            H_theta_pad = H_theta_pad_l
            H_theta_pad_inv = H_theta_pad_inv_l
        elif train_flag == 'w_l':
            H_theta_pad = H_theta_pad_l*0.1+H_theta_pad_w*0.9
            H_theta_pad_inv = H_theta_pad_inv_l*0.1+H_theta_pad_inv_w*0.9
        
        
        
        # affine preditct mask_t, mask_a
        grid = F.affine_grid(H_theta_pad[:,:2,:], m_a.shape, align_corners=True)
        pred_m_a = torch.clamp(F.grid_sample(m_a, grid, align_corners=True), 0.0, 1.0)

        grid = F.affine_grid(H_theta_pad_inv[:,:2,:], m_a.shape, align_corners=True)
        pred_m_t = torch.clamp(F.grid_sample(m_t, grid, align_corners=True), 0.0, 1.0)

    
        # return pred_diff, pred_diff_inv, \
        #         pred_diff_fake_score, pred_diff_score, \
        #         H_theta_pad[:,:2,:], H_theta_nopad[:,:2,:], H_theta_pad_inv[:,:2,:], H_theta_nopad_inv[:,:2,:], \
        #         pred_m_a,  pred_m_t
        
        return pred_diff, pred_diff_inv, \
                H_theta_pad[:,:2,:], H_theta_nopad[:,:2,:], H_theta_pad_inv[:,:2,:], H_theta_nopad_inv[:,:2,:], \
                pred_m_a,  pred_m_t

    def forward_base(self, xt, xa, m_a, m_t, pts_t, train_flag='w'):
        b,c,h,w = xt.shape

        if random.random() < 0.3:
            temp_scale = random.uniform(0.5, 0.8)
            xt_re = F.interpolate(xt, scale_factor=temp_scale, mode='bilinear', align_corners=True)
            xa_re = F.interpolate(xa, scale_factor=temp_scale, mode='bilinear', align_corners=True)
        else:
            xt_re = xt.clone()
            xa_re = xa.clone()

        ft, _ = self.extract_features(xt_re)
        fa, _ = self.extract_features(xa_re)

        pad_size = self.pad_size
        # homography 
        f_ta = torch.cat([ft, fa], dim=1)

        # get the grid points diff 
        features_diff = self.feature_diff(f_ta)
        pred_diff = self.diff_estimator(features_diff)

        # get the homography matrix(no pad) from pd
        if train_flag == 'l':
            pts_t = pts_t.view(b,-1,2)
            pts_diff = pred_diff.view(b,-1,2)
            pts_diff_scale = pts_diff.clone()
            pts_diff_scale[:,:,:1] = pts_diff[:,:,:1]*640
            pts_diff_scale[:,:,1:] = pts_diff[:,:,1:]*480

            pts_p = pts_t + pts_diff_scale
            H_nopad = self.ransac_batch_sim(pts_p, pts_t)
            H_theta_nopad_l = H_cv2torch_batch_sim(H_nopad, w, h)

        else:
            temp_pad_diff = torch.zeros_like(pred_diff)
            H_theta_nopad_w = self.diff2homography_grid(torch.cat([pred_diff, temp_pad_diff], dim=1))
            H_theta_nopad_w = H_theta_nopad_w.view(-1,2,3)

        
        if train_flag == 'w':
            H_theta_nopad = H_theta_nopad_w

        elif train_flag == 'l':
            H_theta_nopad = H_theta_nopad_l

        elif train_flag == 'w_l':
            H_theta_nopad = H_theta_nopad_l*0.1+H_theta_nopad_w*0.9


        


        # get the homography matirx(with pad) from pd
        if train_flag == 'l':
            H_pad = self.ransac_batch_sim(pts_p+pad_size, pts_t+pad_size)
            H_theta_pad_l = H_cv2torch_batch_sim(H_pad, w+2*pad_size, h+2*pad_size)

        else:
            temp_pad_diff = torch.ones_like(pred_diff)
            H_theta_pad_w = self.diff2homography_grid(torch.cat([pred_diff, temp_pad_diff], dim=1))
            H_theta_pad_w = H_theta_pad_w.view(-1,2,3)


        
        if train_flag == 'w':
            H_theta_pad = H_theta_pad_w

        elif train_flag == 'l':
            H_theta_pad = H_theta_pad_l

        elif train_flag == 'w_l':
            H_theta_pad = H_theta_pad_l*0.1+H_theta_pad_w*0.9

        
        
        
        # affine preditct mask_t, mask_a
        grid = F.affine_grid(H_theta_pad[:,:2,:], m_a.shape, align_corners=True)
        pred_m_a = torch.clamp(F.grid_sample(m_a, grid, align_corners=True), 0.0, 1.0)

        
        return pred_diff, H_theta_pad[:,:2,:], H_theta_nopad[:,:2,:], pred_m_a

    def forward(self, xt, xa, m_a, m_t, pts_t, train_flag='w', flag_cycle=False):
        if flag_cycle:
            return self.forward_cycle(xt, xa, m_a, m_t, pts_t, train_flag)
        else:
            return self.forward_base(xt, xa, m_a, m_t, pts_t, train_flag)

    def predict_H(self, xt, xa):
        b,c,h,w = xt.shape
        ft, ft_mask = self.extract_features(xt)
        fa, fa_mask = self.extract_features(xa)

        pad_size = self.pad_size
        # homography 
        f_ta = torch.cat([ft, fa], dim=1)
        # get the grid points diff 
        features_diff = self.feature_diff(f_ta)
        pred_diff = self.diff_estimator(features_diff)

        # features_diff = features_diff.view(b, -1)
        # pred_diff_score = self.diff_score_estimator(torch.cat([features_diff, pred_diff], dim=1))
        
        return pred_diff, pred_diff

    def predict_visual(self, xs, xt):
        b,c,h,w = xs.shape
        fs, fs_mask, list_featuremaps1 = self.extract_visual(xs)
        ft, ft_mask, list_featuremaps2 = self.extract_visual(xt)
        # homography 
        fs_t = torch.cat([fs, ft], dim=1)
        features_diff = self.feature_diff(fs_t)
        pred_diff = self.diff_estimator(features_diff)
        features_diff = features_diff.view(b, -1)
        pred_diff_score = self.diff_score_estimator(torch.cat([features_diff, pred_diff], dim=1))
        
        return pred_diff, pred_diff_score, fs_mask, ft_mask, list_featuremaps1, list_featuremaps2


    @classmethod
    def from_name(cls, model_name, override_params=None, pad_size=400, len_diff=1024):
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, pad_size=pad_size, len_diff=len_diff)

