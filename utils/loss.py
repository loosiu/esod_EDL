# # Loss functions
# # Copyright (c) Alibaba, Inc. and its affiliates.

# ############ EDL ############

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.general import bbox_iou, box_iou, wh_iou, xywh2xyxy
from utils.torch_utils import is_parallel, time_synchronized


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)
            # BCEobj = FocalLoss(BCEobj, g)
        # else:
        #     BCEobj = QFocalLoss(BCEobj, gamma=1.5, alpha=0.5)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'anchor_grid', 'stride':
            setattr(self, k, getattr(det, k))
        self.neg_anchor_iou_thres = 0.7
        self.pos_anchor_iou_thres = 0.15
        self.pos_anchor_num = 4
        self.lpixl_critreia = None

    def __call__(self, p, targets, imgsz=None, masks=None, m_weights=None):  # predictions, targets, model
        p_det, p_seg = p
        offsets = []
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        if p_det is not None and p_det[0] is not None and p_det[1] is not None:  # stupid
            # ta = time_synchronized()
            if isinstance(p_det, tuple):
                p, offsets = p_det
                tcls, tbox, indices, anchors = self.build_patch_targets(offsets, targets, imgsz)  # targets
            else:
                p = p_det
                tcls, tbox, indices, anchors = self.build_targets(p, targets)
            # print(f'build_targets: {time_synchronized() - ta:.3f}s.')

            # Losses
            for i, pi in enumerate(p):  # layer index, layer predictions
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
    
                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
    
                    # Regression
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss
    
                    # Objectness
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
    
                    # Classification
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE
    
                    # Append targets to text file
                    # with open('targets.txt', 'a') as file:
                    #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    
                obji = self.BCEobj(pi[..., 4].clamp_(-9.21, 9.21), tobj)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        # bs = tobj.shape[0]  # batch size
        bs = p_seg[0].shape[0] if p_seg is not None else tobj.shape[0]
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
            
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj'] * 0.5 #(0.5 if (len(offsets) and len(offsets[0]) > bs) else 1.)   # adaoff: 0.178
        lcls *= self.hyp['cls']
        
        if masks is not None and p_seg is not None:
            assert len(p_seg) == 1
            lpixl, larea, ldist = self.compute_loss_seg(p_seg[0], masks, targets, imgsz=imgsz, weight=m_weights)

        loss = (lbox + lobj + lcls) * 1.0 + (lpixl + larea + ldist) * 0.2
        loss_items = torch.cat((lbox, lobj, lcls, lpixl, larea, ldist, loss)).detach()
        return loss * bs, loss_items

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h), 0~1
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter shape(nt_,7), [bi, ci, xc, yc, w, h, ai]

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
  
    def build_patch_targets(self, patch_offsets, targets, imgsz):  # for fast-mode, fixed patch division
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        dtype, device = targets.dtype, targets.device
        tcls, tbox, indices, anch = [], [], [], []
        bs, _, height, width = imgsz
        
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)
        bi_ = torch.arange(patch_offsets[0].shape[0], device=device)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=device).float() * g  # offsets

        for i in range(self.nl):
            patch_off = patch_offsets[i]
            anchors = self.anchors[i]
            r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
            gain[2:6] = torch.tensor([width, height, width, height], dtype=dtype) / (8 * r)  # TODO: from 4 to 32
            # grid_w, grid_h = patch_off[0, [3, 4]] - patch_off[0, [1, 2]]
            grid_wh = patch_off[:1, [3, 4]] - patch_off[:1, [1, 2]]

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter, shape(nt_, 7)

                tb, txc, tyc = t[:, [0, 2, 3]].chunk(3, dim=1)  # shape(n,1)
                pb, px1, py1, px2, py2 = (patch_off.T).chunk(5, dim=0)  # shape(1,m)
                contained = (tb == pb) & (txc > px1 - g) & (txc < px2 - g) & (tyc > py1 - g) & (tyc < py2 - g)  # shape(n,m)
                ti, pj = torch.nonzero(contained).T  # i-th target is contained within j-th patch
                t = t[ti]  # shape(n,7)
                
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = grid_wh - gxy  # inverse
                j, k = ((gxy - gxy.floor() < g) & (gxy > 0.-g)).T
                l, m = ((gxi - gxi.floor() < g) & (gxi > 1.-g)).T
                # j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                
                t[:, 0] = bi_[pj]  # converted batch-indices
                t[:, 2:4] -= patch_off[pj, 1:3]  # converted xc, yc (minus px1, py1)

                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # assert ((gj >= 0) & (gj <= grid_wh[0,1] - 1) & (gi >= 0) & (gi <= grid_wh[0,0] - 1)).all()
            # indices.append((b, a, gj.clamp_(0, grid_wh[0,1] - 1), gi.clamp_(0, grid_wh[0,0] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    # def compute_loss_seg(self, p, masks, targets, weight=None):
    #     dtype, device = targets.dtype, targets.device
    #     bs, nc, ny, nx = masks.shape
    #     assert nc == 1
    #     lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), \
    #                           torch.zeros(1, device=device)
        
    #     # weight = None
    #     lpixl += F.binary_cross_entropy_with_logits(p, masks, weight=weight)

    #     nt = targets.shape[0]
    #     if nt:  # number of targets
    #         pass

    #         # larea += self.dice_loss(p, masks)
    #         # ldist += self.sigmoid_focal_loss(p, masks) * 20
            
    #         # larea += self.quality_dice_loss(p, masks, weight=weight)
    #         # ldist += self.sigmoid_quality_focal_loss(p, masks, weight=weight) * 20

    
    #     return lpixl, larea, ldist
    
    # Segmentation loss dispatch:
    #   shape[1] == 1 : legacy BCE heat (Segmenter,[1])
    #   shape[1] == 2 : legacy EDL (bg,obj evidence logits) → Dirichlet  (Segmenter,[2])
    #   shape[1] == 4 : Full-TMC DUAL — ch0-1 = Heat Dirichlet evidence, ch2-3 = EDL Dirichlet evidence.
    #                   L_seg = lam_h · EDL(α_h, mask) + lam_e · EDL(α_e, mask) + lam_a · EDL(α_a, mask)
    #                   where α_a is the combined Dirichlet from binary TMC DS_Combin.
    #                   Weights via env: ESOD_DUAL_LAM_H (default 1.0), ESOD_DUAL_LAM_E (default 1.0),
    #                                    ESOD_DUAL_LAM_A (default 1.0).  Set LAM_A=0 to disable
    #                                    supervision on the combined view (run DS only at inference).
    def compute_loss_seg(self, p, masks, targets, imgsz=None, weight=None):
        device = targets.device
        bs, nc_t, ny, nx = masks.shape
        assert nc_t == 1, "GT mask는 1채널(0/1)이어야 함"

        Cch = p.shape[1]
        if Cch == 1:
            lpixl = F.binary_cross_entropy_with_logits(
                p, masks, weight=weight, reduction='mean'
            ).reshape(1)
        elif Cch == 2:
            lpixl = self._compute_edl_pixl(p, masks, weight)
        elif Cch == 3:
            # 3-B variants: ch0 = BCE heat logit, ch1-2 = EDL Dirichlet evidence
            # L = lam_h * BCE(ch0) + lam_e * EDL(ch1-2). No combined-α supervision (Heat not Dirichlet here).
            import os as _os
            lam_h = float(_os.environ.get('ESOD_DUAL_LAM_H', '1.0'))
            lam_e = float(_os.environ.get('ESOD_DUAL_LAM_E', '1.0'))

            # 3-C role-separation: SOFT split — both branches receive the FULL Gaussian mask,
            # but per-pixel loss weights emphasize each branch's specialty region.
            # This addresses Park MEH (ICLR'23) "different supervision per attribute" while
            # avoiding the data-starvation failure mode of hard zero-out splits (which on
            # VisDrone gave EDL only ~1.6% positive pixels, collapsing vacuity training).
            #
            # ESOD_ROLE_SEP=1            → activate soft role-separation
            # ESOD_HARD_THRESH_PX (32)   → object pixel-area threshold (32² = 1024 px², COCO AP_s)
            # ESOD_ROLE_SEP_ALPHA (1.0)  → emphasis strength
            #     weight_heat = base_weight · (1 + α · is_easy_region)   # easy regions weighted (1+α)×
            #     weight_edl  = base_weight · (1 + α · is_hard_region)   # hard regions weighted (1+α)×
            # α=0 degenerates to 3-B-a; α=1 → 2× emphasis; α→∞ approaches hard-split limit.
            if _os.environ.get('ESOD_ROLE_SEP', '0') in ('1', 'true', 'True'):
                thresh_px = float(_os.environ.get('ESOD_HARD_THRESH_PX', '32'))
                area_thresh_px2 = thresh_px ** 2
                alpha_emph = float(_os.environ.get('ESOD_ROLE_SEP_ALPHA', '1.0'))
                is_easy, is_hard = self._build_easy_hard_regions(
                    masks, targets, imgsz=imgsz, area_thresh_px2=area_thresh_px2
                )
                base_w = weight if weight is not None else torch.ones_like(masks)
                weight_heat = base_w * (1.0 + alpha_emph * is_easy)
                weight_edl  = base_w * (1.0 + alpha_emph * is_hard)
                # Heat trains on FULL gaussian mask with easy-region emphasis
                l_heat = F.binary_cross_entropy_with_logits(
                    p[:, 0:1], masks, weight=weight_heat, reduction='mean'
                )
                # EDL trains on FULL gaussian mask with hard-region emphasis
                l_edl = self._compute_edl_pixl(p[:, 1:3], masks, weight_edl).squeeze(0)
                # Stats for monitoring (saved into class attribute for later inspection)
                ComputeLoss._role_sep_stats = {
                    'frac_easy_px': float(is_easy.mean()),
                    'frac_hard_px': float(is_hard.mean()),
                    'frac_both_px': float((is_easy * is_hard).mean()),
                    'alpha_emph':   alpha_emph,
                    'avg_w_heat':   float(weight_heat.mean()),
                    'avg_w_edl':    float(weight_edl.mean()),
                }
            else:
                l_heat = F.binary_cross_entropy_with_logits(
                    p[:, 0:1], masks, weight=weight, reduction='mean'
                )
                l_edl = self._compute_edl_pixl(p[:, 1:3], masks, weight).squeeze(0)
            lpixl = (lam_h * l_heat + lam_e * l_edl).reshape(1)
        elif Cch == 4:
            import os as _os
            fusion_mode = _os.environ.get('ESOD_FUSION_MODE', 'dempster').strip().lower()

            if fusion_mode == 'gating':
                # 3-B-b: SegmenterWithGating output
                #   ch 0   = BCE heat logit       → BCE(ch0, mask)
                #   ch 1-2 = EDL Dirichlet bg/obj → EDL(ch1-2, mask)
                #   ch 3   = gating logit         → NO direct supervision on α itself.
                # Auxiliary fusion supervision (lam_f * BCE(F, mask)) gives α an
                # indirect gradient signal — α is shaped to make F match the mask, but
                # is never explicitly told what value to take (preserves CBAM-spirit
                # "no direct attention supervision"). Without this, α has zero gradient
                # because patch-selection uses .detach() and is non-differentiable.
                lam_h = float(_os.environ.get('ESOD_DUAL_LAM_H', '1.0'))
                lam_e = float(_os.environ.get('ESOD_DUAL_LAM_E', '1.0'))
                lam_f = float(_os.environ.get('ESOD_DUAL_LAM_F', '0.5'))

                l_heat = F.binary_cross_entropy_with_logits(
                    p[:, 0:1], masks, weight=weight, reduction='mean'
                )
                l_edl  = self._compute_edl_pixl(p[:, 1:3], masks, weight).squeeze(0)

                if lam_f > 0:
                    # Build F = α·H + (1-α)·V_norm WITHOUT detach so gradient flows to α.
                    from models.common import _view_from_2ch_evidence, normalize_vacuity
                    heat_p = p[:, 0:1].sigmoid()
                    view_e = _view_from_2ch_evidence(p[:, 1:3])
                    v_norm = normalize_vacuity(view_e['u']).unsqueeze(1)
                    alpha_g = p[:, 3:4].sigmoid()
                    F_fused = (alpha_g * heat_p + (1.0 - alpha_g) * v_norm).clamp(1e-6, 1.0 - 1e-6)
                    l_f = F.binary_cross_entropy(F_fused, masks, weight=weight, reduction='mean')
                else:
                    l_f = torch.zeros(1, device=device).squeeze(0)

                lpixl = (lam_h * l_heat + lam_e * l_edl + lam_f * l_f).reshape(1)
            else:
                # 3-A Full TMC: ch 0-1 = Heat Dirichlet, ch 2-3 = EDL Dirichlet
                lam_h = float(_os.environ.get('ESOD_DUAL_LAM_H', '1.0'))
                lam_e = float(_os.environ.get('ESOD_DUAL_LAM_E', '1.0'))
                lam_a = float(_os.environ.get('ESOD_DUAL_LAM_A', '1.0'))

                heat_raw = p[:, 0:2]
                edl_raw  = p[:, 2:4]

                l_heat = self._compute_edl_pixl(heat_raw, masks, weight).squeeze(0)
                l_edl  = self._compute_edl_pixl(edl_raw,  masks, weight).squeeze(0)

                if lam_a > 0:
                    # Build combined Dirichlet α_a via binary TMC DS_Combin, then apply EDL loss to it.
                    # (TMC TPAMI 2022 multi-task loss: sum of per-view + combined-view EDL loss.)
                    from models.common import _view_from_2ch_evidence, tmc_ds_combine_binary
                    v_h = _view_from_2ch_evidence(heat_raw)
                    v_e = _view_from_2ch_evidence(edl_raw)
                    combined = tmc_ds_combine_binary(v_h, v_e, K=2)
                    alpha_a = combined['alpha']                          # [B,2,H,W]
                    l_a = self._compute_edl_pixl_from_alpha(alpha_a, masks, weight).squeeze(0)
                else:
                    l_a = torch.zeros(1, device=device).squeeze(0)

                lpixl = (lam_h * l_heat + lam_e * l_edl + lam_a * l_a).reshape(1)
        else:
            raise ValueError(f"Unsupported segmenter channels: {Cch}")

        larea = torch.zeros(1, device=device)
        ldist = torch.zeros(1, device=device)
        return lpixl, larea, ldist

    def _compute_edl_pixl_from_alpha(self, alpha, masks, weight=None):
        """EDL Dirichlet pixel loss given α directly (no softplus). For combined α_a from TMC."""
        assert alpha.shape[1] == 2, f"EDL pixl-from-alpha expects K=2, got {alpha.shape}"
        device = alpha.device
        S = alpha.sum(dim=1, keepdim=True)
        y = masks
        one_hot = torch.cat([1.0 - y, y], dim=1)

        probs = alpha / S
        err = (one_hot - probs) ** 2
        var = alpha * (S - alpha) / (S ** 2 * (S + 1))
        edl_mse = (err + var).sum(dim=1)

        alpha_tilde = one_hot + (1.0 - one_hot) * alpha
        kl = self._kl_divergence_dirichlet(alpha_tilde)

        annealing_coef = min(1.0, getattr(ComputeLoss, '_edl_epoch', 1) / 10.0)
        edl_loss = edl_mse + annealing_coef * kl
        if weight is not None:
            edl_loss = edl_loss * weight.squeeze(1)
        return edl_loss.mean().reshape(1)

    def _build_easy_hard_regions(self, masks, targets, imgsz, area_thresh_px2=1024.0):
        """3-C: Split GT objects into easy (large, area ≥ thresh) and hard (small, area < thresh)
        by object pixel area, then build per-pixel binary region masks in the mask-resolution space.

        Args:
            masks   : [B, 1, H_m, W_m] — the GT mask (Gaussian/SAM-smoothed) used for supervision.
            targets : [N, 6] = [batch_idx, cls, xc, yc, w, h] all normalized to [0, 1].
            imgsz   : (B, C, H, W) of the input image tensor (used to convert normalized targets
                       to pixel area).
            area_thresh_px2 : object pixel-area threshold for the easy/hard split. Default 32² = 1024
                       (COCO AP_s vs AP_m boundary). Configurable via env ESOD_HARD_THRESH_PX (squared).

        Returns:
            is_easy : [B, 1, H_m, W_m] binary — 1 where pixel falls inside any EASY (large) target.
            is_hard : [B, 1, H_m, W_m] binary — 1 where pixel falls inside any HARD (small) target.

        Pixels with no overlapping target → both is_easy=0 and is_hard=0 (background).
        Pixels inside overlapping easy+hard objects → both =1 (heat & EDL both supervise that pixel).
        """
        B, _, H_m, W_m = masks.shape
        img_h, img_w = int(imgsz[2]), int(imgsz[3])

        is_easy = torch.zeros_like(masks)
        is_hard = torch.zeros_like(masks)

        if targets is None or len(targets) == 0:
            return is_easy, is_hard

        # Pixel area per target: w_norm * h_norm * img_w * img_h
        areas_px = (targets[:, 4] * img_w) * (targets[:, 5] * img_h)
        is_easy_obj = areas_px >= area_thresh_px2     # [N] bool

        for n in range(targets.shape[0]):
            bi = int(targets[n, 0])
            xc, yc = float(targets[n, 2]), float(targets[n, 3])
            w_n, h_n = float(targets[n, 4]), float(targets[n, 5])
            # Convert to mask-resolution rectangle [x1, y1, x2, y2]
            x_c_m, y_c_m = xc * W_m, yc * H_m
            w_m_obj, h_m_obj = w_n * W_m, h_n * H_m
            x1 = max(0, int(x_c_m - w_m_obj / 2.0))
            x2 = min(W_m, int(x_c_m + w_m_obj / 2.0) + 1)
            y1 = max(0, int(y_c_m - h_m_obj / 2.0))
            y2 = min(H_m, int(y_c_m + h_m_obj / 2.0) + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            if bool(is_easy_obj[n]):
                is_easy[bi, 0, y1:y2, x1:x2] = 1.0
            else:
                is_hard[bi, 0, y1:y2, x1:x2] = 1.0
        return is_easy, is_hard

    def _compute_edl_pixl(self, p, masks, weight=None):
        """EDL Dirichlet pixel loss on a 2-ch (bg,obj) evidence-logits tensor.

        학습 목표: 배경/객체 모두 evidence를 높여 vacuity↓; 학습이 부족한 영역(작은 객체 등)
        은 자연스럽게 vacuity↑ → 추론 시 탐지 후보로 사용 가능.
        """
        assert p.shape[1] == 2, f"EDL pixl expects 2-ch logits, got {p.shape}"
        device = p.device

        evidence = F.softplus(p)              # [B,2,H,W] >= 0
        alpha = evidence + 1.0                # Dirichlet params
        S = alpha.sum(dim=1, keepdim=True)    # [B,1,H,W]

        y = masks                             # [B,1,H,W]
        one_hot = torch.cat([1.0 - y, y], dim=1)  # [B,2,H,W]

        probs = alpha / S
        err = (one_hot - probs) ** 2
        var = alpha * (S - alpha) / (S ** 2 * (S + 1))
        edl_mse = (err + var).sum(dim=1)      # [B,H,W]

        alpha_tilde = one_hot + (1.0 - one_hot) * alpha
        kl = self._kl_divergence_dirichlet(alpha_tilde)

        annealing_coef = min(1.0, getattr(ComputeLoss, '_edl_epoch', 1) / 10.0)
        edl_loss = edl_mse + annealing_coef * kl

        if weight is not None:
            edl_loss = edl_loss * weight.squeeze(1)

        return edl_loss.mean().reshape(1)

    @staticmethod
    def _kl_divergence_dirichlet(alpha):
        """KL(Dir(alpha) || Dir(1,...,1)) per pixel. alpha: [B, K, H, W]"""
        K = alpha.shape[1]
        S = alpha.sum(dim=1)  # [B, H, W]
        kl = torch.lgamma(S) - torch.lgamma(torch.tensor(float(K), device=alpha.device)) \
             - torch.lgamma(alpha).sum(dim=1) \
             + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(1)))).sum(dim=1)
        return kl

    @staticmethod
    def dice_loss(inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

    @staticmethod
    def quality_dice_loss(inputs, targets, weight=None, gamma: float = 2):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        if weight is not None:
            weight = weight.flatten(1)
            inputs = inputs * weight
            targets = targets * weight

        numerator = 2 * (inputs - targets).abs().sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = (numerator + 1) / (denominator + 1)
        return loss.mean()

    @staticmethod
    def sigmoid_quality_focal_loss(inputs, targets, weight=None, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction="none")
        loss = ce_loss * ((prob - targets).abs() ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

### graph ###
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math

# from utils.general import bbox_iou, box_iou, wh_iou, xywh2xyxy
# from utils.torch_utils import is_parallel, time_synchronized


# def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
#     # return positive, negative label smoothing BCE targets
#     return 1.0 - 0.5 * eps, 0.5 * eps


# class BCEBlurWithLogitsLoss(nn.Module):
#     # BCEwithLogitLoss() with reduced missing label effects.
#     def __init__(self, alpha=0.05):
#         super(BCEBlurWithLogitsLoss, self).__init__()
#         self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
#         self.alpha = alpha

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         pred = torch.sigmoid(pred)  # prob from logits
#         dx = pred - true  # reduce only missing label effects
#         # dx = (pred - true).abs()  # reduce missing label and false label effects
#         alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
#         loss *= alpha_factor
#         return loss.mean()


# class FocalLoss(nn.Module):
#     # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         super(FocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'  # required to apply FL to each element

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         # p_t = torch.exp(-loss)
#         # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

#         # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
#         pred_prob = torch.sigmoid(pred)  # prob from logits
#         p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = (1.0 - p_t) ** self.gamma
#         loss *= alpha_factor * modulating_factor

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss


# class QFocalLoss(nn.Module):
#     # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         super(QFocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'  # required to apply FL to each element

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)

#         pred_prob = torch.sigmoid(pred)  # prob from logits
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = torch.abs(true - pred_prob) ** self.gamma
#         loss *= alpha_factor * modulating_factor

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss


# class ComputeLoss:
#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         super(ComputeLoss, self).__init__()
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters

#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls = FocalLoss(BCEcls, g)
#             # BCEobj = FocalLoss(BCEobj, g)
#         # else:
#         #     BCEobj = QFocalLoss(BCEobj, gamma=1.5, alpha=0.5)

#         det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
#         self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
#         for k in 'na', 'nc', 'nl', 'anchors', 'anchor_grid', 'stride':
#             setattr(self, k, getattr(det, k))
#         self.neg_anchor_iou_thres = 0.7
#         self.pos_anchor_iou_thres = 0.15
#         self.pos_anchor_num = 4
#         self.lpixl_critreia = None

#     def __call__(self, p, targets, imgsz=None, masks=None, m_weights=None):  # predictions, targets, model
#         p_det, p_seg = p
#         offsets = []
#         device = targets.device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
#         if p_det is not None and p_det[0] is not None and p_det[1] is not None:  # stupid
#             # ta = time_synchronized()
#             if isinstance(p_det, tuple):
#                 p, offsets = p_det
#                 tcls, tbox, indices, anchors = self.build_patch_targets(offsets, targets, imgsz)  # targets
#             else:
#                 p = p_det
#                 tcls, tbox, indices, anchors = self.build_targets(p, targets)
#             # print(f'build_targets: {time_synchronized() - ta:.3f}s.')

#             # Losses
#             for i, pi in enumerate(p):  # layer index, layer predictions
#                 b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#                 tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
    
#                 n = b.shape[0]  # number of targets
#                 if n:
#                     ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
    
#                     # Regression
#                     pxy = ps[:, :2].sigmoid() * 2. - 0.5
#                     pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                     pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                     iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
#                     lbox += (1.0 - iou).mean()  # iou loss
    
#                     # Objectness
#                     tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
    
#                     # Classification
#                     if self.nc > 1:  # cls loss (only if multiple classes)
#                         t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
#                         t[range(n), tcls[i]] = self.cp
#                         lcls += self.BCEcls(ps[:, 5:], t)  # BCE
    
#                     # Append targets to text file
#                     # with open('targets.txt', 'a') as file:
#                     #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    
#                 obji = self.BCEobj(pi[..., 4].clamp_(-9.21, 9.21), tobj)
#                 lobj += obji * self.balance[i]  # obj loss
#                 if self.autobalance:
#                     self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
#         # bs = tobj.shape[0]  # batch size
#         bs = p_seg[0].shape[0] if p_seg is not None else tobj.shape[0]
#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
            
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj'] * 0.5 #(0.5 if (len(offsets) and len(offsets[0]) > bs) else 1.)   # adaoff: 0.178
#         lcls *= self.hyp['cls']
        
#         if masks is not None and p_seg is not None:
#             assert len(p_seg) == 1
#             lpixl, larea, ldist = self.compute_loss_seg(p_seg[0], masks, targets, weight=m_weights)
        
#         loss = (lbox + lobj + lcls) * 1.0 + (lpixl + larea + ldist) * 0.2
#         loss_items = torch.cat((lbox, lobj, lcls, lpixl, larea, ldist, loss)).detach()
#         return loss * bs, loss_items

#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h), 0~1
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)

#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                             ], device=targets.device).float() * g  # offsets

#         for i in range(self.nl):
#             anchors = self.anchors[i]
#             gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter shape(nt_,7), [bi, ci, xc, yc, w, h, ai]

#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#                 l, m = ((gxi % 1. < g) & (gxi > 1.)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0

#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices

#             # Append
#             a = t[:, 6].long()  # anchor indices
#             indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class

#         return tcls, tbox, indices, anch
  
#     def build_patch_targets(self, patch_offsets, targets, imgsz):  # for fast-mode, fixed patch division
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         dtype, device = targets.dtype, targets.device
#         tcls, tbox, indices, anch = [], [], [], []
#         bs, _, height, width = imgsz
        
#         gain = torch.ones(7, device=device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, shape(na,nt,7)
#         bi_ = torch.arange(patch_offsets[0].shape[0], device=device)

#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                             ], device=device).float() * g  # offsets

#         for i in range(self.nl):
#             patch_off = patch_offsets[i]
#             anchors = self.anchors[i]
#             r = (2 ** (i - 1)) if self.nl == 4 else 2 ** i
#             gain[2:6] = torch.tensor([width, height, width, height], dtype=dtype) / (8 * r)  # TODO: from 4 to 32
#             # grid_w, grid_h = patch_off[0, [3, 4]] - patch_off[0, [1, 2]]
#             grid_wh = patch_off[:1, [3, 4]] - patch_off[:1, [1, 2]]

#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter, shape(nt_, 7)

#                 tb, txc, tyc = t[:, [0, 2, 3]].chunk(3, dim=1)  # shape(n,1)
#                 pb, px1, py1, px2, py2 = (patch_off.T).chunk(5, dim=0)  # shape(1,m)
#                 contained = (tb == pb) & (txc > px1 - g) & (txc < px2 - g) & (tyc > py1 - g) & (tyc < py2 - g)  # shape(n,m)
#                 ti, pj = torch.nonzero(contained).T  # i-th target is contained within j-th patch
#                 t = t[ti]  # shape(n,7)
                
#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = grid_wh - gxy  # inverse
#                 j, k = ((gxy - gxy.floor() < g) & (gxy > 0.-g)).T
#                 l, m = ((gxi - gxi.floor() < g) & (gxi > 1.-g)).T
#                 # j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#                 # l, m = ((gxi % 1. < g) & (gxi > 1.)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
                
#                 t[:, 0] = bi_[pj]  # converted batch-indices
#                 t[:, 2:4] -= patch_off[pj, 1:3]  # converted xc, yc (minus px1, py1)

#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

#             else:
#                 t = targets[0]
#                 offsets = 0

#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices

#             # Append
#             a = t[:, 6].long()  # anchor indices
#             # assert ((gj >= 0) & (gj <= grid_wh[0,1] - 1) & (gi >= 0) & (gi <= grid_wh[0,0] - 1)).all()
#             # indices.append((b, a, gj.clamp_(0, grid_wh[0,1] - 1), gi.clamp_(0, grid_wh[0,0] - 1)))  # image, anchor, grid indices
#             indices.append((b, a, gj, gi))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class

#         return tcls, tbox, indices, anch

#     # def compute_loss_seg(self, p, masks, targets, weight=None):
#     #     dtype, device = targets.dtype, targets.device
#     #     bs, nc, ny, nx = masks.shape
#     #     assert nc == 1
#     #     lpixl, larea, ldist = torch.zeros(1, device=device), torch.zeros(1, device=device), \
#     #                           torch.zeros(1, device=device)
        
#     #     # weight = None
#     #     lpixl += F.binary_cross_entropy_with_logits(p, masks, weight=weight)

#     #     nt = targets.shape[0]
#     #     if nt:  # number of targets
#     #         pass

#     #         # larea += self.dice_loss(p, masks)
#     #         # ldist += self.sigmoid_focal_loss(p, masks) * 20
            
#     #         # larea += self.quality_dice_loss(p, masks, weight=weight)
#     #         # ldist += self.sigmoid_quality_focal_loss(p, masks, weight=weight) * 20

    
#     #     return lpixl, larea, ldist
    
#     # EDL 기반 Segmentation Loss
#     def compute_loss_seg(self, p, masks, targets, weight=None):
#         """
#         p: [B,2,H,W]  (bg,obj evidence logits)
#         masks: [B,1,H,W] (0/1 binary GT mask)
#         """
#         device = targets.device
#         bs, nc_t, ny, nx = masks.shape
#         assert nc_t == 1, "GT mask는 1채널(0/1)이어야 함"
#         assert p.shape[1] == 2, f"EDL 사용 시 pred channels must be 2, got {p.shape}"

#         # evidence / alpha
#         evidence = F.softplus(p)          # [B,2,H,W]
#         alpha = evidence + 1.0
#         S = alpha.sum(dim=1, keepdim=True)  # [B,1,H,W]
#         probs = alpha / S                  # expected prob

#         y = masks
#         y2 = torch.cat([1.0 - y, y], dim=1)  # [B,2,H,W]

#         # expected CE
#         ce = -(y2 * (probs.clamp_min(1e-8)).log()).sum(dim=1, keepdim=True)  # [B,1,H,W]
#         lpixl = ce.mean()  # <-- 0-dim 이 됨

#         # evidence regularizer (과확신 억제 + 붕괴 방지)
#         e_bg = evidence[:, 0:1]
#         e_obj = evidence[:, 1:2]
#         reg = (y * e_bg + (1.0 - y) * e_obj).mean()
#         lpixl = lpixl + 0.01 * reg

#         # ===== 핵심: cat에 들어가도록 (1,) shape로 강제 =====
#         lpixl = lpixl.reshape(1)

#         larea = torch.zeros(1, device=device)
#         ldist = torch.zeros(1, device=device)
#         return lpixl, larea, ldist

#     @staticmethod
#     def dice_loss(inputs, targets):
#         """
#         Compute the DICE loss, similar to generalized IOU for masks
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#         """
#         inputs = inputs.sigmoid().flatten(1)
#         targets = targets.flatten(1)
#         numerator = 2 * (inputs * targets).sum(-1)
#         denominator = inputs.sum(-1) + targets.sum(-1)
#         loss = 1 - (numerator + 1) / (denominator + 1)
#         return loss.mean()

#     @staticmethod
#     def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
#         """
#         Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#             alpha: (optional) Weighting factor in range (0,1) to balance
#                     positive vs negative examples. Default = -1 (no weighting).
#             gamma: Exponent of the modulating factor (1 - p_t) to
#                 balance easy vs hard examples.
#         Returns:
#             Loss tensor
#         """
#         prob = inputs.sigmoid()
#         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#         p_t = prob * targets + (1 - prob) * (1 - targets)
#         loss = ce_loss * ((1 - p_t) ** gamma)

#         if alpha >= 0:
#             alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#             loss = alpha_t * loss

#         return loss.mean()

#     @staticmethod
#     def quality_dice_loss(inputs, targets, weight=None, gamma: float = 2):
#         """
#         Compute the DICE loss, similar to generalized IOU for masks
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#         """
#         inputs = inputs.sigmoid().flatten(1)
#         targets = targets.flatten(1)
#         if weight is not None:
#             weight = weight.flatten(1)
#             inputs = inputs * weight
#             targets = targets * weight

#         numerator = 2 * (inputs - targets).abs().sum(-1)
#         denominator = inputs.sum(-1) + targets.sum(-1)
#         loss = (numerator + 1) / (denominator + 1)
#         return loss.mean()

#     @staticmethod
#     def sigmoid_quality_focal_loss(inputs, targets, weight=None, alpha: float = 0.25, gamma: float = 2):
#         """
#         Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#         Args:
#             inputs: A float tensor of arbitrary shape.
#                     The predictions for each example.
#             targets: A float tensor with the same shape as inputs. Stores the binary
#                     classification label for each element in inputs
#                     (0 for the negative class and 1 for the positive class).
#             alpha: (optional) Weighting factor in range (0,1) to balance
#                     positive vs negative examples. Default = -1 (no weighting).
#             gamma: Exponent of the modulating factor (1 - p_t) to
#                 balance easy vs hard examples.
#         Returns:
#             Loss tensor
#         """
#         prob = inputs.sigmoid()
#         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weight, reduction="none")
#         loss = ce_loss * ((prob - targets).abs() ** gamma)

#         if alpha >= 0:
#             alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#             loss = alpha_t * loss

#         return loss.mean()