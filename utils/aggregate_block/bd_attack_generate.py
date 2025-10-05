# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.

import sys, logging
sys.path.append('../../')
import imageio
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from utils.bd_img_transform.lc import labelConsistentAttack
from utils.bd_img_transform.blended import blendedImageAttack
from utils.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger
from utils.bd_img_transform.sig import sigTriggerAttack
from utils.bd_img_transform.SSBA import SSBA_attack_replace_version
from utils.bd_img_transform.ftrojann import ftrojann_version
from utils.bd_label_transform.backdoor_label_transform import *
from torchvision.transforms import Resize
from utils.bd_img_transform.ctrl import ctrl


class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list
    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img

class convertNumpyArrayToFloat32(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np_img_float32.astype(np.float32)
npToFloat32 = convertNumpyArrayToFloat32()

class clipAndConvertNumpyArrayToUint8(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np.clip(np_img_float32, 0, 255).astype(np.uint8)
npClipAndToUint8 = clipAndConvertNumpyArrayToUint8()

def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''
    
    # Get attack type from either args.attack or args.attack_type
    attack = getattr(args, 'attack', None) or getattr(args, 'attack_type', 'badnet')

    if attack in ['badnet',]:


        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(args.patch_mask_path))
        ) 

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif attack == 'blended':

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_train_blended_alpha)), True), # 0.1,
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_test_blended_alpha)), True), # 0.1,
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

    elif attack == 'sig':
        trans = sigTriggerAttack(
            delta=args.sig_delta,
            f=args.sig_f,
        )
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif attack in ['ssba', 'SSBA']:
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_train_replace_imgs_path)  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif attack in ['SSBA_version2','ssba_version2']:
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_train_replace_imgs_path)  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
    elif args.attack in ['label_consistent']:
        add_trigger = labelConsistentAttack(reduced_amplitude=args.reduced_amplitude)
        add_trigger_func = add_trigger.poison_from_indices
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
                replace_images=np.load(args.attack_train_replace_imgs_path)  # '../data/cifar10_SSBA/train.npy'
            ), True),
            (add_trigger_func, False),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            # (SSBA_attack_replace_version(
            #     replace_images=np.load(args.attack_test_replace_imgs_path)  # '../data/cifar10_SSBA/test.npy'
            # ), True),
            (add_trigger_func, False),
            (npClipAndToUint8,False),
            (Image.fromarray,False),
        ])

    elif attack == 'lowFrequency':

        triggerArray = np.load(args.lowFrequencyPatternPath)

        if len(triggerArray.shape) == 4:
            logging.info("Get lowFrequency trigger with 4 dimension, take the first one")
            triggerArray = triggerArray[0]
        elif len(triggerArray.shape) == 3:
            pass
        elif len(triggerArray.shape) == 2:
            triggerArray =  np.stack((triggerArray,)*3, axis=-1)
        else:
            raise ValueError("lowFrequency trigger shape error, should be either 2 or 3 or 4")

        logging.info("Load lowFrequency trigger with shape {}".format(triggerArray.shape))

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = triggerArray,
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = triggerArray,
            ), True),
            (npClipAndToUint8,False),
            (Image.fromarray, False),
        ])
    elif attack == "ctrl":
        train_bd_transform = ctrl(args, train=True)
        test_bd_transform = ctrl(args, train=False)

    elif attack == "ftrojann":
        bd_transform = ftrojann_version(YUV=args.YUV, channel_list=args.channel_list, window_size=args.window_size, magnitude=args.magnitude, pos_list=args.pos_list)

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, False),
            ]
        )

    elif attack == "wanet":
        # WaNet warping-based backdoor transform
        import torch
        import torch.nn.functional as F
        s = getattr(args, 's', 0.5)
        k = getattr(args, 'k', 4)
        grid_rescale = getattr(args, 'grid_rescale', 1)
        input_height = getattr(args, 'input_height', 32)
        input_width = getattr(args, 'input_width', 32)
        device = getattr(args, 'device', 'cpu')
        # Generate warping grid
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.interpolate(ins, size=(input_height, input_width), mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)
        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        grid = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid = torch.clamp(grid, -1, 1)
        grid = grid.squeeze(0)
        # Define the transform
        def wanet_transform(img, target=None, image_serial_id=None):
            import torchvision.transforms.functional as TF
            img_tensor = TF.to_tensor(img).unsqueeze(0)
            warped = F.grid_sample(img_tensor, grid.unsqueeze(0), align_corners=True)
            warped_img = TF.to_pil_image(warped.squeeze(0))
            return warped_img
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (wanet_transform, True),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (wanet_transform, True),
        ])
    elif attack == "trojannn":
        # TrojanNN attack using mask from mask_path
        
        # Load the mask image
        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])
        
        mask_image = trans(Image.open(args.mask_path))
        
        # Simple TrojanTrigger class for basic additive trigger
        class TrojanTrigger(object):
            def __init__(self, target_image):
                self.target_image = target_image.astype(float)

            def __call__(self, img, target=None, image_serial_id=None):
                return self.add_trigger(img)

            def add_trigger(self, img):
                return np.clip((self.target_image + img.astype(float)).astype("uint8"), 0, 255)
        
        bd_transform = TrojanTrigger(mask_image)

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8, False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8, False),
            (Image.fromarray, False),
        ])

    elif attack == "inputaware":
        # InputAware attack uses dynamic neural network-based backdoor generation
        # during training, so we use identity transforms here
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (npClipAndToUint8, False),
            (Image.fromarray, False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (npClipAndToUint8, False),
            (Image.fromarray, False),
        ])

    elif attack == "refool":
        # Refool attack - need to import and use RefoolTrigger
        import os
        from attack.refool import RefoolTrigger
        
        # Load reflection images
        reflection_img_list = []
        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2]),
            np.array,
        ])
        
        for img_name in os.listdir(args.r_adv_img_folder_path):
            full_img_path = os.path.join(args.r_adv_img_folder_path, img_name)
            reflection_img = Image.open(full_img_path)
            reflection_img_list.append(trans(reflection_img))
            reflection_img.close()
        
        bd_transform = RefoolTrigger(
            reflection_img_list,
            args.img_size[0],
            args.img_size[1], 
            args.ghost_rate,
            alpha_t=args.alpha_t,
            offset=args.offset,
            sigma=args.sigma,
            ghost_alpha=args.ghost_alpha,
        )
        
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
        ])
        
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
        ])

    return train_bd_transform, test_bd_transform

def bd_attack_label_trans_generate(args):
    '''
    # idea : use args to choose which backdoor label transform you want
    from args generate backdoor label transformation

    '''
    if args.attack_label_trans == 'all2one':
        target_label = int(args.attack_target)
        bd_label_transform = AllToOne_attack(target_label)
    elif args.attack_label_trans == 'all2all':
        bd_label_transform = AllToAll_shiftLabelAttack(
            int(1 if "attack_label_shift_amount" not in args.__dict__ else args.attack_label_shift_amount), int(args.num_classes)
        )

    return bd_label_transform
