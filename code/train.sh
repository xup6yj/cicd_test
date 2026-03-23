#!/bin/bash

python main.py \
--model CSFDAFAPUNetV2 \
--in_channel 1 \
--out_channel 2 \
--init_ch 32 \
--criterion UnifiedFocalLoss \
--num_epochs 300 \
--batch_size 8 \
--queue_length 1000 \
--samples_per_volume 8 \
--cv 2 \
--dataset ncct111_gt-1_5fold \
--lr 2e-5 \
--uf_loss_weight 1.0 \
--bd_loss \
--bd_loss_weight 0.2 \
--patch_x 128 \
--patch_y 128 \
--patch_z 8 \
--patch_overlap_x 0 \
--patch_overlap_y 0 \
--patch_overlap_z 2 \
--cgm \
--cgm_weight 0.2 \
--curriculum \
--cpt_dir cpts/ncct111_gt-1_5fold/CSFDAFAPUNetV2.cpts/sub \
--log_dir logs/ncct111_gt-1_5fold/CSFDAFAPUNetV2.logs/sub \
# --continue_training \
# --continue_epoch 100 \
# --continue_batch_done 73416 \
# --continue_cpt_path /adc/research/CTAIS/cpts/AISD_5fold_spm/DMGAPDAFAUNet.cpts/DMGAPDAFAUNet_CGM-0.2_in1_out2_init32_UFL-w0.5-d0.6-g0.2_bdloss-w0.2_lr2e-05_kaiming-init_300epochs_bs8_p128-128-8_o0-0-2_AISD_5fold_spm_curr_cv4/cv4_epoch266_batch73416_e266.cpt \
# --cpt_path /adc/research/CTAIS/cpts/AISD_5fold_spm/UNetPP.cpts/UNetPP_CGM-0.2_in1_out2_init32_UFL-w0.5-d0.6-g0.2_bdloss-w0.2_lr2e-05_kaiming-init_300epochs_bs8_p128-128-8_o0-0-2_AISD_5fold_spm_curr_cv2/cv2_epoch264_batch73128_e264.cpt
# --exponential_lr \
# --weight_init cr \
# --cr_cpt_path cr.cpts/ResUNet_in1_out1_init32_lr0.0002_kaiming-init_300epochs_bs8_p128-128-8_cr/epoch200_batch388000_e200.cpt
