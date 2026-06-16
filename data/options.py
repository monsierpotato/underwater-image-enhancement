"""
Underwater Image Restoration — Training Options
Replaces the CIDNet low-light options with UWIR-specific arguments.
"""

import argparse


def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def option():
    parser = argparse.ArgumentParser(description='UWIR — Physics-Guided Underwater Image Restoration')

    # ------------------------------------------------------------------
    # Core training hyper-parameters
    # ------------------------------------------------------------------
    parser.add_argument('--batchSize',    type=int,   default=16,
                        help='Training mini-batch size')
    parser.add_argument('--cropSize',     type=int,   default=256,
                        help='Resize target size for training images (height=width)')
    parser.add_argument('--nEpochs',      type=int,   default=200,
                        help='Total number of training epochs')
    parser.add_argument('--start_epoch',  type=int,   default=0,
                        help='Starting epoch (> 0 resumes from checkpoint)')
    parser.add_argument('--resume',        type=str,   default='',
                        help='Path to a checkpoint .pth to resume training from '
                             '(overrides --start_epoch logic). Example: '
                             '--resume ./checkpoints/run/epoch_0040.pth')
    parser.add_argument('--snapshots',    type=int,   default=10,
                        help='Save a checkpoint every N epochs')
    parser.add_argument('--lr',           type=float, default=1e-4,
                        help='Initial learning rate (Adam)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Adam weight decay')
    parser.add_argument('--gpu_mode',     type=_str2bool, default=True)
    parser.add_argument('--num_gpus',     type=int,       default=1,
                        help='Number of GPUs to use via DataParallel (1 = single GPU)')
    parser.add_argument('--shuffle',      type=_str2bool, default=True)
    parser.add_argument('--threads',      type=int,   default=4,
                        help='DataLoader worker threads (keep low to avoid OOM on Kaggle; 2-4 recommended)')

    # ------------------------------------------------------------------
    # Learning-rate scheduler
    # ------------------------------------------------------------------
    parser.add_argument('--cos_restart',       type=_str2bool, default=False,
                        help='Use CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False,
                        help='Use cyclic cosine restart variant')
    parser.add_argument('--scheduler_step',    type=int,   default=30,
                        help='StepLR step size in epochs')
    parser.add_argument('--scheduler_gamma',   type=float, default=0.5,
                        help='StepLR decay factor')
    parser.add_argument('--warmup_epochs',     type=int,   default=0,
                        help='Number of linear warm-up epochs')
    parser.add_argument('--start_warmup',      type=_str2bool, default=False,
                        help='Enable warm-up at the start of training')

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Stop training if val SSIM does not improve '
                             'for this many epochs (proposal §4.5)')

    # ------------------------------------------------------------------
    # Physics front-end
    # ------------------------------------------------------------------
    parser.add_argument('--prior_method', type=str, default='udcp',
                        choices=['udcp', 'gdcp', 'multi_prior'],
                        help='Transmission-map / background-light estimation method')
    parser.add_argument('--guided_filter_radius', type=int, default=40,
                        help='Radius for the guided image filter used to '
                             'refine the transmission map')
    parser.add_argument('--guided_filter_eps',    type=float, default=1e-3,
                        help='Regularisation epsilon for the guided filter')

    # ------------------------------------------------------------------
    # Model / ablation variant
    # ------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='unet_5ch',
                        choices=[
                            # U-Net (no pretrained encoder)
                            'unet_3ch', 'unet_4ch_t', 'unet_4ch_b', 'unet_5ch',
                            # ResNet-50 encoder
                            'resnet_3ch', 'resnet_4ch_t', 'resnet_4ch_b', 'resnet_5ch',
                            # MobileNetV3-Large encoder
                            'mobilenet_3ch', 'mobilenet_4ch_t', 'mobilenet_4ch_b', 'mobilenet_5ch',
                        ],
                        help=(
                            'Model variant (backbone_channels):\n'
                            '  Channels: 3ch=RGB only | 4ch_t=RGB+t(x) | 4ch_b=RGB+B | 5ch=RGB+t(x)+B\n'
                            '  Backbones: unet | resnet (ResNet-50) | mobilenet (MobileNetV3-Large)'
                        ))
    parser.add_argument('--pretrained_backbone', type=_str2bool, default=True,
                        help='Load ImageNet-pretrained weights for ResNet / MobileNet encoders')
    parser.add_argument('--backbone', type=str, default='unet',
                        choices=['unet', 'resnet', 'mobilenet'],
                        help='Refinement backbone architecture (inferred from --model if unset)')

    # ------------------------------------------------------------------
    # Loss weights  λ1·L_pixel + λ2·L_perceptual + λ3·L_SSIM
    # ------------------------------------------------------------------
    parser.add_argument('--L1_weight',         type=float, default=1.0,
                        help='λ1 — pixel-wise MAE loss weight')
    parser.add_argument('--perceptual_weight',  type=float, default=1.0,
                        help='λ2 — VGG-16 perceptual loss weight')
    parser.add_argument('--SSIM_weight',        type=float, default=0.0,
                        help='λ3 — SSIM loss weight')

    # ------------------------------------------------------------------
    # Training dataset paths
    # ------------------------------------------------------------------
    parser.add_argument('--data_train_euvp',
                        type=str, default='./datasets/EUVP',
                        help='Root of EUVP release (primary training corpus)')
    parser.add_argument('--euvp_subset',
                        type=str, default='all',
                        help=(
                            'EUVP sub-set(s) to use for training.\n'
                            '  "all"              — all three subsets (notebook default)\n'
                            '  "underwater_imagenet" | "underwater_dark" | "underwater_scenes"\n'
                            '  Comma-separated for multiple: "underwater_dark,underwater_scenes"'
                        ))
    parser.add_argument('--data_train_uieb',
                        type=str, default='./datasets/UIEB',
                        help='Root of UIEB release (supplementary training)')

    # ------------------------------------------------------------------
    # Validation / evaluation input paths
    # ------------------------------------------------------------------
    parser.add_argument('--data_val_uieb',
                        type=str, default='./datasets/UIEB/test/input',
                        help='UIEB test-90 input images')
    parser.add_argument('--data_val_ufo120',
                        type=str, default='./datasets/UFO120/test/lrd',
                        help='UFO-120 test input images')
    parser.add_argument('--data_val_euvp',
                        type=str, default='./datasets/EUVP/Paired/underwater_imagenet/validation',
                        help='EUVP unpaired validation images (no GT available)')
    parser.add_argument('--data_val_u45',
                        type=str, default='./datasets/U45',
                        help='U45 no-reference evaluation images')

    # ------------------------------------------------------------------
    # Validation / evaluation ground-truth paths
    # ------------------------------------------------------------------
    parser.add_argument('--data_valgt_uieb',
                        type=str, default='./datasets/UIEB/test/reference',
                        help='UIEB test-90 reference (ground-truth) images')
    parser.add_argument('--data_valgt_ufo120',
                        type=str, default='./datasets/UFO120/test/hr',
                        help='UFO-120 test ground-truth images')
    parser.add_argument('--data_valgt_euvp',
                        type=str, default='',
                        help='EUVP ground-truth (empty: EUVP validation is unpaired, no GT)')

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    parser.add_argument('--val_folder',
                        type=str, default='./results/',
                        help='Directory for saving validation output images')
    parser.add_argument('--checkpoint_dir',
                        type=str, default='./checkpoints/',
                        help='Directory for saving model checkpoints')

    # ------------------------------------------------------------------
    # Misc / reproducibility
    # ------------------------------------------------------------------
    parser.add_argument('--seed',       type=int,       default=42,
                        help='Global random seed for reproducibility')
    parser.add_argument('--grad_clip',  type=_str2bool, default=True,
                        help='Enable gradient clipping to stabilise training')
    parser.add_argument('--grad_detect', type=_str2bool, default=False,
                        help='Enable anomaly detection (slow; use for debugging only)')

    # ------------------------------------------------------------------
    # Dataset selector (controls which loader is used in train.py)
    # ------------------------------------------------------------------
    parser.add_argument('--dataset', type=str, default='euvp',
                        choices=['euvp', 'uieb', 'ufo120', 'euvp+uieb'],
                        help=(
                            'Training dataset:\n'
                            '  euvp       — EUVP only  (primary, proposal §4.5)\n'
                            '  uieb       — UIEB only\n'
                            '  ufo120     — UFO-120 only\n'
                            '  euvp+uieb  — Combined EUVP + UIEB'
                        ))

    return parser
