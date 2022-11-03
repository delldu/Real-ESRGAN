"""Image/Video Zoom Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import todos

from . import rrdbnet

import pdb


def get_zoom2x_model():
    """Create model."""

    model_path = "models/image_zoom2x.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = rrdbnet.RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    todos.model.load(model, checkpoint)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_zoom2x.torch"):
        model.save("output/image_zoom2x.torch")

    return model, device


def get_zoom4x_model():
    """Create model."""

    model_path = "models/image_zoom4x.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = rrdbnet.RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_zoom4x.torch"):
        model.save("output/image_zoom4x.torch")

    return model, device


def zoom2x_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_zoom2x_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.shape
        orig_tensor = todos.data.resize_tensor(input_tensor, 2 * H, 2 * W)
        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def zoom4x_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_zoom4x_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.shape
        orig_tensor = todos.data.resize_tensor(input_tensor, 4 * H, 4 * W)
        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
