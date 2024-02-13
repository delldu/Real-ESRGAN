#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>

#include "tensor.h"
#include "rrdbnet.h"

#define DEFAULT_OUTPUT "output"

void help(char *cmd)
{
    printf("Usage: %s [option] image_files\n", cmd);
    printf("    -h, --help                   Display this help.\n");
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda 0, default: 1).\n");

    printf("    -o, --output                 Output dir (default: %s).\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char **argv)
{
    int optc;
    int option_index = 0;
    int device_no = 1;
    char *output_dir = (char *)DEFAULT_OUTPUT;

    struct option long_opts[] = {
        { "help", 0, 0, 'h'},
        { "device", 1, 0, 'd'},
        { "output", 1, 0, 'o'},
        { 0, 0, 0, 0}
    };

    if (argc <= 1)
        help(argv[0]);
    
    while ((optc = getopt_long(argc, argv, "h d: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'd':
            device_no = atoi(optarg);
            break;
        case 'o':
            output_dir=optarg;
            break;
        case 'h':   // help
        default:
            help(argv[0]);
            break;
        }
    }

    make_dir(output_dir);

    struct RRDBNet net;

    net.set_device(device_no); // 0 -- cpu, 1 -- cuda
    net.load("models/anime4x.gguf", "");
    net.start_engine();
    // net.dump();

    // MS -- Modify Section ?
    char *p, output_fname[512];
    TENSOR *input_tensor, *output_tensor;
    TENSOR *tensor_argv[1];

    for (int i = optind; i < argc; i++) {
        p = strrchr(argv[i], '/');
        p = (!p) ? argv[i] : p + 1;
        snprintf(output_fname, sizeof(output_fname) - 1, "%s/%s", output_dir, p);

        printf("%s ---> %s\n", argv[i], output_fname);

        input_tensor = tensor_load_image(argv[i], 0 /*without alpha*/);
        check_tensor(input_tensor);

        tensor_argv[0] = input_tensor;
        output_tensor = net.execute_forward(ARRAY_SIZE(tensor_argv), tensor_argv);
        check_tensor(output_tensor);
        if (output_tensor) {
            tensor_saveas_image(output_tensor, 0 /*batch 0*/, output_fname);
            tensor_destroy(output_tensor);
        }

        tensor_destroy(input_tensor);
    }

    net.stop_engine();
    return 0;
}
