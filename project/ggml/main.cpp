#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "rrdbnet.h"
#include "tensor.h"

#define DEFAULT_OUTPUT "output"

void help(char* cmd)
{
    printf("Usage: %s [option] image_files\n", cmd);
    printf("    -h, --help                   Display this help.\n");
    printf("    -m, --model <n|a>            Set model (n -- normal, a -- animate, default: n).\n");
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda 0, default: 1).\n");
    printf("    -o, --output                 Output dir (default: %s).\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;
    int device_no = 1;
    char model = 'n';
    char* output_dir = (char*)DEFAULT_OUTPUT;

    struct option long_opts[] = {
         { "help", 0, 0, 'h' },
         { "model", 1, 0, 'm' }, 
         { "device", 1, 0, 'd' }, 
         { "output", 1, 0, 'o' }, 
         { 0, 0, 0, 0 }
    };

    if (argc <= 1)
        help(argv[0]);

    while ((optc = getopt_long(argc, argv, "h m: d: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'm':
            model = optarg[0];
            break;
        case 'd':
            device_no = atoi(optarg);
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'h': // help
        default:
            help(argv[0]);
            break;
        }
    }

    if (model == 'a')
        return anime4x_predict(device_no, argc - optind, &argv[optind], output_dir);
    else
        return zoom4x_predict(device_no, argc - optind, &argv[optind], output_dir);
}
