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
from torch.nn import functional as F

import todos
from . import rrdbnet
import pdb


def get_image_zoom2x_model():
    """Create model."""
    model = rrdbnet.RRDBNet("image_zoom2x", num_block=23, num_grow_ch=32, scale=2)
    model = todos.model.ResizePadModel(model, scale=2)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type): # reset model to cpu
        model = model.float()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_zoom2x.torch"):
        model.save("output/image_zoom2x.torch")

    return model, device


def get_image_zoom4x_model():
    """Create model."""

    model = rrdbnet.RRDBNet("image_zoom4x", num_block=23, num_grow_ch=32, scale=4)
    model = todos.model.ResizePadModel(model, scale=4)

    # num_block=23, scale=4

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type): # reset model to cpu
        model = model.float()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_zoom4x.torch"):
        model.save("output/image_zoom4x.torch")

    return model, device


def get_image_smooth4x_model():
    """Create model."""

    model = rrdbnet.RRDBNet("image_smooth4x", num_block=23, num_grow_ch=32, scale=4)
    model = todos.model.ResizePadModel(model, scale=4)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type): # reset model to cpu
        model = model.float()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_smooth4x.torch"):
        model.save("output/image_smooth4x.torch")

    return model, device


def get_image_anime4x_model():
    """Create model."""

    model = rrdbnet.RRDBNet("image_anime4x", num_block=6, num_grow_ch=32, scale=4)
    model = todos.model.ResizePadModel(model, scale=4)

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type): # reset model to cpu
        model = model.float()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_anime4x.torch"):
        model.save("output/image_anime4x.torch")

    return model, device

# def get_video_anime4x_model():
#     """Create model."""
#     model = rrdbnet.SRVGGNetCompact("video_anime4x", num_conv=16) # realesr-animevideov3.pth

#     device = todos.model.get_device()
#     model = model.to(device)
#     model.eval()
#     if 'cpu' in str(device.type): # reset model to cpu
#         model = model.float()

#     print(f"Running on {device} ...")
#     # make sure model good for C/C++
#     model = torch.jit.script(model)
#     # https://github.com/pytorch/pytorch/issues/52286
#     torch._C._jit_set_profiling_executor(False)
#     # C++ Reference
#     # torch::jit::getProfilingMode() = false;                                                                                                             
#     # torch::jit::setTensorExprFuserEnabled(false);

#     todos.data.mkdir("output")
#     if not os.path.exists("output/video_anime4x.torch"):
#         model.save("output/video_anime4x.torch")

#     return model, device


def get_image_denoise4x_model():
    """Create model."""
    model = rrdbnet.SRVGGNetDenoise()
    model = todos.model.ResizePadModel(model, scale=4)

    # 'realesr-general-x4v3.pth', 'realesr-general-wdn-x4v3.pth'

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    if 'cpu' in str(device.type): # reset model to cpu
        model = model.float()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_denoise4x.torch"):
        model.save("output/image_denoise4x.torch")

    return model, device


def image_zoom2x_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_image_zoom2x_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # if 'cuda' in str(device.type):
        #     input_tensor = input_tensor.half()

        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.shape
        orig_tensor = todos.data.resize_tensor(input_tensor, 2 * H, 2 * W)
        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def image_zoom4x_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_image_zoom4x_model()
    
    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # if 'cuda' in str(device.type):
        #     input_tensor = input_tensor.half()

        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.shape
        orig_tensor = todos.data.resize_tensor(input_tensor, 4 * H, 4 * W)
        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def image_anime4x_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_image_anime4x_model()
    
    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # if 'cuda' in str(device.type):
        #     input_tensor = input_tensor.half()

        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.shape
        orig_tensor = todos.data.resize_tensor(input_tensor, 4 * H, 4 * W)
        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


def image_smooth4x_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_image_smooth4x_model()
    
    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # if 'cuda' in str(device.type):
        #     input_tensor = input_tensor.half()

        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.shape
        orig_tensor = todos.data.resize_tensor(input_tensor, 4 * H, 4 * W)
        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()


# def video_anime4x_predict(input_files, output_dir):
#     # Create directory to store result
#     todos.data.mkdir(output_dir)

#     # load model
#     model, device = get_video_anime4x_model()
    
#     # load files
#     image_filenames = todos.data.load_files(input_files)

#     # start predict
#     progress_bar = tqdm(total=len(image_filenames))
#     for filename in image_filenames:
#         progress_bar.update(1)

#         # orig input
#         input_tensor = todos.data.load_tensor(filename)
#         if 'cuda' in str(device.type):
#             input_tensor = input_tensor.half()

#         predict_tensor = todos.model.forward(model, device, input_tensor)
#         output_file = f"{output_dir}/{os.path.basename(filename)}"

#         B, C, H, W = input_tensor.shape
#         orig_tensor = todos.data.resize_tensor(input_tensor, 4 * H, 4 * W)
#         todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
#     todos.model.reset_device()

def image_denoise4x_predict(input_files, noise_strength, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_image_denoise4x_model()
    
    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)

        # if 'cuda' in str(device.type):
        #     input_tensor = input_tensor.half()

        with torch.no_grad():
            predict_tensor = model(input_tensor.to(device), noise_strength)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        B, C, H, W = input_tensor.shape
        orig_tensor = todos.data.resize_tensor(input_tensor, 4 * H, 4 * W)
        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()