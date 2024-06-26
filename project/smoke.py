# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import os
import torch
import image_zoom

import argparse
import todos
import pdb


def test_input_shape():
    import time
    import random
    from tqdm import tqdm

    print("Test input shape ...")

    model, device = image_zoom.get_image_anime4x_model() # zoom4x_model()

    N = 100
    B, C, H, W = 1, 3, model.MAX_H, model.MAX_W

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        h = random.randint(-16, 16)
        w = random.randint(-16, 16)
        x = torch.randn(B, C, H + h, W + w)
        # print("x: ", x.size())
        # if 'cuda' in str(device.type):
        #     x = x.half()

        start_time = time.time()
        with torch.no_grad():
            y = model(x.to(device))
        if 'cpu' not in str(device):
            torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")


def run_bench_mark():
    print("Run benchmark ...")

    model, device = image_zoom.get_image_zoom4x_model()
    N = 100
    B, C, H, W = 1, 3, model.MAX_H, model.MAX_W

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(N):
            image = torch.randn(B, C, H, W)
            # if 'cuda' in str(device.type):
            #     image = image.half()

            with torch.no_grad():
                y = model(image.to(device))
            if 'cpu' not in str(device):
                torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")


def export_onnx_model(model_name="zoom4x"):
    import onnx
    import onnxruntime
    from onnxsim import simplify
    import onnxoptimizer

    print("Export onnx model ...")

    # 1. Run torch model
    if model_name == "zoom4x":
        model, device = image_zoom.get_image_zoom4x_model()
    if model_name == "zoom2x":
        model, device = image_zoom.get_image_zoom2x_model()
    if model_name == "anime4x":
        model, device = image_zoom.get_image_anime4x_model()

    B, C, H, W = 1, 3, 512, 512 # model.MAX_H, model.MAX_W

    dummy_input = torch.randn(B, C, H, W).to(device)
    # if 'cuda' in str(device.type):
    #     dummy_input = dummy_input.half()

    with torch.no_grad():
        dummy_output = model(dummy_input)
    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "input" ]
    output_names = [ "output" ]
    dynamic_axes = { 
        'input' : {2: 'height', 3: 'width'}, 
        'output' : {2: 'height', 3: 'width'} 
    }

    if model_name == "zoom4x":
        onnx_filename = "output/image_zoom4x.onnx"
    if model_name == "zoom2x":
        onnx_filename = "output/image_zoom2x.onnx"
    if model_name == "anime4x":
        onnx_filename = "output/image_anime4x.onnx"

    torch.onnx.export(model, dummy_input, onnx_filename, 
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # 3. Check onnx model file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    if model_name == "zoom2x":
        # !!! onnxsim not support pixel_unshuffle !!! 
        # onnx_model, check = simplify(onnx_model)
        # # assert check, "Simplified ONNX model could not be validated"
        # print(onnx.helper.printable_graph(onnx_model.graph))
        pass
    else:
        onnx_model, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
    onnx_model = onnxoptimizer.optimize(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    onnx.save(onnx_model, onnx_filename)

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = {input_names[0]: to_numpy(dummy_input) }
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.01, atol=0.01)

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")
    todos.model.reset_device()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoke Test')
    parser.add_argument('-s', '--shape_test', action="store_true", help="test shape")
    parser.add_argument('-b', '--bench_mark', action="store_true", help="test benchmark")
    parser.add_argument('-e', '--export_onnx', action="store_true", help="export onnx model")
    args = parser.parse_args()

    if args.shape_test:
        test_input_shape()
    if args.bench_mark:
        run_bench_mark()
    if args.export_onnx:
        # export DEVICE=cpu for OOM of cuda
        export_onnx_model("zoom4x")
        # export_onnx_model("zoom2x")
        # export_onnx_model("anime4x")
    
    if not (args.shape_test or args.bench_mark or args.export_onnx):
        parser.print_help()
