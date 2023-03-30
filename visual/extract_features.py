#!/usr/bin/env python
"""Script to extract ResNet features from video frames."""
import argparse
from typing import Any, Tuple

import h5py
from overrides import overrides
import torch
import torch.nn
import torch.utils.data
import torchvision
from tqdm import tqdm

# from c3d import C3D
# from i3d import I3D
from dataset import SarcasmDataset

from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls
from collections import OrderedDict 
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch.hub import load_state_dict_from_url

# noinspection PyUnresolvedReferences
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152


def save_resnet_features() -> None:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = SarcasmDataset(transform=transforms)

    # resnet = pretrained_resnet152().to(DEVICE)

    # class Identity(torch.nn.Module):
    #     @overrides
    #     def forward(self, input_: torch.Tensor) -> torch.Tensor:
    #         return input_
    
    class Identity(ResNet):
        def __init__(self,output_layer,*args):
            self.output_layer = output_layer
            super().__init__(*args)
            
            self._layers = []
            for l in list(self._modules.keys()):
                self._layers.append(l)
                if l == output_layer:
                    break
            self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))
    
        def _forward_impl(self, x):
            for l in self._layers:
                x = self.layers[l](x)
    
            return x
    
        def forward(self, x):
            return self._forward_impl(x)
    def new_resnet(
        arch: str,
        outlayer: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
    ) -> Identity:
    
        '''model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
            'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
            'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
            'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        }'''

        model = Identity(outlayer, block, layers, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            model.load_state_dict(state_dict)
        return model
        
    resnet = new_resnet('resnet50','layer4',Bottleneck, [3, 4, 6, 3],True,True)  # Trick to avoid computing the fc1000 layer, as we don't need it here.

    # with h5py.File(SarcasmDataset.features_file_path("resnet", "res5c"), "w") as res5c_features_file, \
    #         h5py.File(SarcasmDataset.features_file_path("resnet", "pool5"), "w") as pool5_features_file:

    for video_id in dataset.video_ids:
        video_frame_count = dataset.frame_count_by_video_id[video_id]
        # res5c_features_file.create_dataset(video_id, shape=(video_frame_count, 2048, 7, 7))
        # pool5_features_file.create_dataset(video_id, shape=(video_frame_count, 2048))

    res5c_output = None

    def avg_pool_hook(_module: torch.nn.Module, input_: Tuple[torch.Tensor], _output: Any) -> None:
        nonlocal res5c_output
        res5c_output = input_[0]

    resnet.avgpool.register_forward_hook(avg_pool_hook)

    total_frame_count = sum(dataset.frame_count_by_video_id[video_id] for video_id in dataset.video_ids)
    with tqdm(total=total_frame_count, desc="Extracting ResNet features") as progress_bar:
        for instance in torch.utils.data.DataLoader(dataset):
            video_id = instance["id"][0]
            print(video_id)
            frames = instance["frames"][0].to(DEVICE)
            print(len(frames))
            # break
            

            batch_size = 32
            avg_pool_merged = torch.Tensor()
            for start_index in range(0, len(frames), batch_size):
                # print(start_index)
                end_index = min(start_index + batch_size, len(frames))
                frame_ids_range = range(start_index, end_index)
                frame_batch = frames[frame_ids_range]
                
                avg_pool_value = resnet(frame_batch)
                
                # res5c_features_file[video_id][frame_ids_range] = res5c_output.cpu()  # noqa
                # pool5_features_file[video_id][frame_ids_range] = avg_pool_value.cpu().data.numpy()
                avg_pool_merged = torch.cat((avg_pool_merged, avg_pool_value), 0)
                progress_bar.update(len(frame_ids_range))
            import numpy as np
            # np.save("../data/features/utterances_final/resnet_pool5_" + str(video_id), avg_pool_merged.cpu().data.numpy())
            np.save("../data/features/context_final/resnet_pool5_" + str(video_id), avg_pool_merged.cpu().data.numpy())



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract video features.")
    parser.add_argument("network", choices=["resnet", "c3d", "i3d"])
    return parser.parse_args()


def main() -> None:
    # save_resnet_features()
    args = parse_args()
    if args.network == "resnet":
        save_resnet_features()
    # elif args.network == "c3d":
    #     save_c3d_features()
    # elif args.network == "i3d":
    #     save_i3d_features()
    else:
        raise ValueError(f"Network type not supported: {args.network}")


if __name__ == "__main__":
    main()
