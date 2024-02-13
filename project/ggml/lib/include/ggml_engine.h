/************************************************************************************
***
***	Copyright 2024 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 30 Jan 2024 11:52:34 PM CST
***
************************************************************************************/

#ifndef _GGML_ENGINE_H
#define _GGML_ENGINE_H

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <nimage/image.h>

// GGML Engine
typedef struct {
    // General
    int device = 0; // 0 -- CPU, 1 -- CUDA 0, 2 -- CUDA 1, ...
    int cpu_threads = 1;
    size_t num_tensors = 0;
    struct ggml_context* context = NULL;

    // Backend
    ggml_backend_t backend = NULL;
    size_t backend_buffer_size = 0;
    ggml_backend_buffer_t backend_buffer = NULL;

    // Model weight
    const char* model_name = "";
    const char* weight_prefix = "";

} GGMLEngine;

typedef struct {
    // ------------------ public --------------------------------------------------------
    void dump();
    void set_device(int device);
    bool load(const char* model_path, const char* prefix);

    bool start_engine();
    TENSOR* execute_forward(int eng_argc, TENSOR* eng_argv[]);
    void stop_engine();

    virtual void create_weight_tensors(struct ggml_context* ctx) { GGML_UNUSED(ctx); }

    virtual void setup_weight_names(char* prefix) { GGML_UNUSED(prefix); }

    virtual struct ggml_tensor* forward(struct ggml_context* ctx, int eng_argc, struct ggml_tensor* eng_argv[])
    {
        GGML_UNUSED(ctx);
        GGML_UNUSED(eng_argc);
        auto x = eng_argv[0];

        return ggml_dup_inplace(ctx, x); // do not use 'return x;' directly !!!
        // return ggml_mul_inplace(ctx, x, x);
        // return ggml_add(ctx, x, x);
    }

    virtual size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE; // 2048
    }

    // ------------------ private ------------------------------------------------------
    GGMLEngine m_ggml_engine = {};

    bool network_init();
    struct ggml_cgraph* build_graph(int eng_argc, TENSOR* eng_argv[]);
} GGMLNetwork;

void dump_ggml_tensor(const char* prefix, struct ggml_tensor* tensor, bool more);
// void set_tensor_name(struct ggml_tensor* tensor, const char* prefix, const char* name);
bool set_tensor_value(struct ggml_tensor* tensor, TENSOR* nt, bool to_backend); // nt -- nimage tensor
TENSOR* get_tensor_value(struct ggml_tensor* tensor, bool from_backend);

#endif // _GGML_ENGINE_H
