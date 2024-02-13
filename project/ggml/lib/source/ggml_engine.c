/************************************************************************************
***
***	Copyright 2024 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 30 Jan 2024 11:53:11 PM CST
***
************************************************************************************/
// ggml-alloc v3
// https://github.com/ggerganov/ggml/pull/72

#include "../include/ggml_engine.h"

#ifdef GGML_CUBLAS
#define GGML_USE_CUBLAS
#endif

#ifdef GGML_METAL
#define GGML_USE_METAL
#endif

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <thread>
#include <vector>

#include <float.h>
// #include <stdlib.h>

#define MAX_ENG_ARGC 8

#define check_avoid(x)                                                                                                 \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            fflush(stdout);                                                                                            \
            fprintf(stderr, "Error: %s != NULL/true (%s:%d)\n", #x, __FILE__, __LINE__);                               \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define check_point(x)                                                                                                 \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            fflush(stdout);                                                                                            \
            fprintf(stderr, "Error: %s != NULL/true (%s:%d)\n", #x, __FILE__, __LINE__);                               \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_POINT(x)                                                                                                 \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            fflush(stdout);                                                                                            \
            fprintf(stderr, "Error: %s != NULL/true (%s:%d)\n", #x, __FILE__, __LINE__);                               \
            return NULL;                                                                                               \
        }                                                                                                              \
    } while (0)

#define for_each_context_tensor(ctx)                                                                                   \
    for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t))

#define MAKE_INPUT_NAME(name, i) snprintf(name, sizeof(name), "ggml_engine_input_%02d", i)

static struct ggml_context* create_context(size_t mem_size, bool static_alloc);
static char* find_model_path(const char* model_name);
static struct ggml_backend* create_backend(int device, int* ok_device);

static bool data_type_valid(ggml_type dtype);
static bool same_data_shape(struct ggml_tensor* tensor, TENSOR* nt);

// The follwing functions is like ggml_get_f32_1d/ggml_set_f32_1d, but much more general
static float get_f32_from_buffer(void* data, ggml_type dtype, size_t i);
static bool set_f32_to_buffer(void* data, ggml_type dtype, size_t i, float value);

static void network_exit(GGMLEngine* eng);

static bool backend_init(GGMLEngine* eng);
static void backend_exit(GGMLEngine* eng);

static bool load_weight_from_gguf(GGMLEngine* eng);

static const char* backend_name(int b);

// --------------------------------------------------------------------------------------
void GGMLNetwork::set_device(int device)
{
    m_ggml_engine.device = device;
}

bool GGMLNetwork::start_engine()
{
    syslog_info("Start Engine (%s:%s) ...", ENGINE_VERSION, RELEASE_DATE);

    check_point(network_init());
    check_point(backend_init(&m_ggml_engine));
    check_point(load_weight_from_gguf(&m_ggml_engine));

    syslog_info("Start Engine OK.");

    return true;
}

void GGMLNetwork::stop_engine()
{
    syslog_info("Stop Engine ...");
    // system("nvidia-smi");

    // compute_exit(&m_ggml_engine);
    backend_exit(&m_ggml_engine);
    network_exit(&m_ggml_engine);

    // system("nvidia-smi");
}

bool GGMLNetwork::network_init()
{
    GGMLEngine* eng = &m_ggml_engine;

    if (eng->context) // do not repeat init ...
        return true;

    // Get num of tensors and memoy size via temp context for more presion
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ggml_tensor_overhead() * eng->num_tensors + 1 * 1024 * 1024,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true, // the tensors no need to be allocated later
        };
        struct ggml_context* temp_context = ggml_init(params);
        check_point(temp_context);
        create_weight_tensors(temp_context);

        eng->num_tensors = 0;
        eng->backend_buffer_size = 10 * 1024 * 1024; // 10M for padding
        for_each_context_tensor(temp_context)
        {
            eng->num_tensors++;
            eng->backend_buffer_size += ggml_nbytes(t);
        }
        ggml_free(temp_context);
    }

    // Create tensor and their memory on device cpu
    {
        size_t mem_size = ggml_tensor_overhead() * eng->num_tensors + eng->backend_buffer_size;
        eng->context = create_context(mem_size, true); // not allocate tensor memory directly and done via backend

        if (eng->context) {
            syslog_info("Network memory %.2f MB (%ld tensors).", eng->backend_buffer_size / (1024.0 * 1024.0),
                eng->num_tensors);

            create_weight_tensors(eng->context);
            setup_weight_names((char *)""); // prefix_name
        }
    }

    check_point(eng->context);
    return true;
}

static void network_exit(GGMLEngine* eng)
{
    check_avoid(eng);
    check_avoid(eng->context);
    ggml_free(eng->context);
}

static bool backend_init(GGMLEngine* eng)
{
    check_point(eng);
    check_point(eng->context);

    // Create backend and backend buffer
    {
        eng->backend = create_backend(eng->device, &eng->device);
        check_point(eng->backend);

        syslog_info("Backend buffer %.2f MB.", eng->backend_buffer_size / (1024.0 * 1024.0));

        // eng->backend_buffer = ggml_backend_alloc_ctx_tensors(eng->context, eng->backend);
        eng->backend_buffer = ggml_backend_alloc_buffer(eng->backend, eng->backend_buffer_size);
        check_point(eng->backend_buffer);
    }

    // Allocate backend memory for tensors ...
    {
        struct ggml_tallocr* alloc = ggml_tallocr_new(eng->backend_buffer);
        check_point(alloc);
        for_each_context_tensor(eng->context) {
            ggml_tallocr_alloc(alloc, t);
        }
        ggml_tallocr_free(alloc);
    }

    // Set CPU threads ...
    {
        eng->cpu_threads = MAX(std::thread::hardware_concurrency() / 2, 1);

        if (ggml_backend_is_cpu(eng->backend)) {
            ggml_backend_cpu_set_n_threads(eng->backend, eng->cpu_threads);
        }
#ifdef GGML_USE_METAL
        if (ggml_backend_is_metal(eng->backend)) {
            ggml_backend_metal_set_n_cb(eng->backend, eng->cpu_threads);
        }
#endif
    }

    return true;
}

static void backend_exit(GGMLEngine* eng)
{
    check_avoid(eng);

    if (eng->backend_buffer != NULL)
        ggml_backend_buffer_free(eng->backend_buffer);
    if (eng->backend != NULL)
        ggml_backend_free(eng->backend);

    eng->backend = NULL;
    eng->backend_buffer = NULL;
    eng->backend_buffer_size = 0;
}

void GGMLNetwork::dump()
{
    check_avoid(m_ggml_engine.context);

    syslog_info("Network information: ");
    syslog_info("  CPU threads: %d", m_ggml_engine.cpu_threads);
    syslog_info("  Backend device number: %d", m_ggml_engine.device);
    syslog_info("  Backend name: %s", ggml_backend_name(m_ggml_engine.backend));
    syslog_info("  Backend buffer: %.2f MB", m_ggml_engine.backend_buffer_size / (1024.0 * 1024.0));
    syslog_info("  Weight model: %s, prefix: %s", m_ggml_engine.model_name, m_ggml_engine.weight_prefix);

    syslog_info("Network Tensors: %d", m_ggml_engine.num_tensors);
    for_each_context_tensor(m_ggml_engine.context) { dump_ggml_tensor("  ", t, false); }
}

bool GGMLNetwork::load(const char* model_name, const char* prefix)
{
    char* model_filename = find_model_path(model_name);
    if (model_filename) {
        free(model_filename);

        m_ggml_engine.model_name = model_name;
        m_ggml_engine.weight_prefix = prefix;

        return true;
    }

    return false;
}

static bool load_weight_from_gguf(GGMLEngine* eng)
{
    check_point(eng);
    check_point(strlen(eng->model_name) > 0); // Skip no-existed model ...

    char* model_filename = NULL;
    struct gguf_context* ctx_gguf = NULL;
    struct {
        struct ggml_context* context;
    } weight;

    // Loading weight
    {
        syslog_info("Loading weight from '%s' with prefix '%s' ...", eng->model_name, eng->weight_prefix);

        model_filename = find_model_path(eng->model_name);
        check_point(model_filename);

        struct gguf_init_params params = {
            /*.no_alloc   =*/false,
            /*.ctx        =*/&weight.context,
        };

        ctx_gguf = gguf_init_from_file(model_filename, params);
        if (! ctx_gguf) {
            syslog_error("Loading gguf file '%s'", model_filename);
            free(model_filename);
        }

        check_point(ctx_gguf);
        check_point(weight.context);
    }

    // Network loading weight ...
    {
        size_t prefix_len = strlen(eng->weight_prefix);
        struct ggml_tensor* destion_tensor = NULL;
        bool backend_is_cpu = true;

        if (ggml_backend_is_cpu(eng->backend)
#ifdef GGML_USE_METAL
            || ggml_backend_is_metal(eng->backend)
#endif
        ) {
            backend_is_cpu = true;
        } else {
            backend_is_cpu = false;
        }

        for_each_context_tensor(weight.context)
        {
            if (memcmp(t->name, eng->weight_prefix, prefix_len) != 0) {
                syslog_debug("Skip '%s' for mismatch '%s' ...", t->name, eng->weight_prefix);
                continue;
            }

            // Real name should be t->name + prefix_len !!!
            destion_tensor = ggml_get_tensor(eng->context, t->name + prefix_len);
            if (destion_tensor == NULL) {
                 // Skip empty name for it maybe gguf whole data
                if (strlen(t->name + prefix_len) > 0)
                    syslog_debug("Skip '%s' for not defined in network ...", t->name + prefix_len);
                continue;
            }
            if (!ggml_are_same_shape(destion_tensor, t)) {
                syslog_error("%s shape mismatch: got [%ld, %ld, %ld, %ld], expected [%ld, %ld, %ld, %ld]",
                    destion_tensor->name, t->ne[0], t->ne[1], t->ne[2], t->ne[3], destion_tensor->ne[0],
                    destion_tensor->ne[1], destion_tensor->ne[2], destion_tensor->ne[3]);
                continue;
            }

            // Loading tensors ...
            if (destion_tensor->type == t->type) { // fast set
                // memcpy(destion_tensor->data, t->data, ggml_nbytes(destion_tensor));
                if (backend_is_cpu) {
                    memcpy(destion_tensor->data, t->data, ggml_nbytes(destion_tensor));
                } else {
                    // cuda requires copy the data directly to device
                    ggml_backend_tensor_set(destion_tensor, t->data, 0, ggml_nbytes(destion_tensor));
                }
            } else { // slow set
                void* temp_data = malloc(ggml_nbytes(destion_tensor));
                check_point(temp_data);
                for (size_t i = 0; i < (size_t)ggml_nelements(destion_tensor); i++) {
                    float value = ggml_get_f32_1d(t, i);
                    set_f32_to_buffer(temp_data, destion_tensor->type, i, value);
                }
                // memcpy(destion_tensor->data, temp_data, ggml_nbytes(destion_tensor));
                if (backend_is_cpu) {
                    memcpy(destion_tensor->data, temp_data, ggml_nbytes(destion_tensor));
                } else {
                    ggml_backend_tensor_set(destion_tensor, temp_data, 0, ggml_nbytes(destion_tensor));
                }
                free(temp_data);
            }
            syslog_debug("Loading %s ... OK", destion_tensor->name);
        }
    }

    // Clean up
    {
        gguf_free(ctx_gguf);
        ggml_free(weight.context);
        free(model_filename);
    }

    return true;
}

struct ggml_cgraph* GGMLNetwork::build_graph(int eng_argc, TENSOR* eng_argv[])
{
    char name[128];
    struct ggml_tensor* ggml_tensor_argv[MAX_ENG_ARGC];

    // GGMLEngine* eng = &m_ggml_engine;
    CHECK_POINT(eng_argc < MAX_ENG_ARGC);

    static size_t buf_size = ggml_tensor_overhead() * this->get_graph_size() + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf.data(),
        /*.no_alloc   =*/true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // Create a temporally context to build the graph
    struct ggml_context* graph_ctx = ggml_init(params0);

    struct ggml_cgraph* gf = ggml_new_graph(graph_ctx);
    // struct ggml_cgraph* gf = ggml_new_graph_custom(graph_ctx, this->get_graph_size(), true);

    for (int i = 0; i < eng_argc; i++) {
        TENSOR *t = eng_argv[i];
        ggml_tensor_argv[i] = ggml_new_tensor_4d(graph_ctx, GGML_TYPE_F32, 
            (int64_t)t->width, (int64_t)t->height, (int64_t)t->chan, (int64_t)t->batch);
        MAKE_INPUT_NAME(name, i);
        ggml_set_name(ggml_tensor_argv[i], name);
        ggml_set_input(ggml_tensor_argv[i]);
    }

    struct ggml_tensor* result = this->forward(graph_ctx, eng_argc, ggml_tensor_argv);
    ggml_set_name(result, "ggml_engine_output");
    ggml_set_output(result);

    ggml_build_forward_expand(gf, result);

    // Delete the temporally context used to build the graph
    ggml_free(graph_ctx);
    return gf;
}


TENSOR* GGMLNetwork::execute_forward(int eng_argc, TENSOR* eng_argv[])
{
    char name[128];
    CHECK_POINT(eng_argc < MAX_ENG_ARGC);
    struct ggml_tensor* x;

    GGMLEngine* eng = &m_ggml_engine;
    ggml_time_init();
    uint64_t start = ggml_time_ms();

    struct ggml_cgraph * gf = NULL;
    ggml_gallocr_t allocr = NULL;
    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(eng->backend));

        gf = build_graph(eng_argc, eng_argv);

        // Create the worst case graph for memory usage estimation
        CHECK_POINT(ggml_gallocr_reserve(allocr, gf));
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        syslog_info("Compute buffer: %.2f M", mem_size/(1024.0*1024.0));
    }

    CheckPoint("time pass: %ld ms", ggml_time_ms() - start);
    // gf = build_graph(eng_argc, eng_argv);

    // Allocate tensors
    CHECK_POINT(ggml_gallocr_alloc_graph(allocr, gf));
    CheckPoint("time pass: %ld ms", ggml_time_ms() - start);

    // Set input value to backend
    for (int i = 0; i < eng_argc; i++) {
        MAKE_INPUT_NAME(name, i);
        x = ggml_graph_get_tensor(gf, name);
        set_tensor_value(x, eng_argv[i], true);
    }

    CheckPoint("time pass: %ld ms, n_threads = %d", ggml_time_ms() - start, eng->cpu_threads);

    // Run the computation
    CHECK_POINT(ggml_backend_graph_compute(eng->backend, gf));

    CheckPoint("time pass: %ld ms", ggml_time_ms() - start);

    if (getenv("DEBUG")) {
       ggml_graph_print(gf);
    }

    // Save output to nt tensor
    struct ggml_tensor* result = ggml_graph_get_tensor(gf, "ggml_engine_output"); // gf->nodes[gf->n_nodes - 1];
    CHECK_POINT(result);
    dump_ggml_tensor("result", result, false);

    TENSOR* output = get_tensor_value(result, true);

    CheckPoint("time pass: %ld ms", ggml_time_ms() - start);

    ggml_gallocr_free(allocr);

    CheckPoint("time pass: %ld ms", ggml_time_ms() - start);

    return output;
}

static char* find_model_path(const char* model_name)
{
    if (access(model_name, F_OK) == 0) {
        syslog_info("Found model '%s'.", model_name);
        return strdup(model_name);
    }

    // Try to find model under modes/
    char filename[512];
    snprintf(filename, sizeof(filename), "models/%s", model_name);
    if (access(filename, F_OK) == 0) {
        syslog_info("Found model '%s'.", filename);
        return strdup(filename);
    }

    syslog_error("Model '%s' NOT Found !!!", model_name);
    return NULL;
}

static struct ggml_backend* create_backend(int device, int* ok_device)
{
#ifdef GGML_USE_CUBLAS
    if (device) {
        struct ggml_backend* backend = ggml_backend_cuda_init(device - 1); // cuda 0 ...
        if (backend) {
            syslog_info("Using CUDA(%d) as backend.", device - 1);
            *ok_device = device;
            return backend;
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (device) {
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
        struct ggml_backend* backend = ggml_backend_metal_init();
        if (backend) {
            syslog_info("Using Metal as backend.");
            *ok_device = 0; // force set to cpu !!!
            return backend;
        }
    }
#endif
    // Fallback to CPU backend
    syslog_info("Using CPU as backend.");
    *ok_device = 0; // force set to cpu !!!
    return ggml_backend_cpu_init();
}

// void set_tensor_name(struct ggml_tensor* tensor, const char* prefix, const char* name)
// {
//     char full_name[1024];
//     snprintf(full_name, sizeof(full_name), "%s%s", prefix, name);
//     ggml_set_name(tensor, full_name);
// }

static const char* backend_name(int b)
{
    if (b == 0)
        return "CPU";
    if (b == 10)
        return "CUDA";
    if (b == 20)
        return "Metal";

    return "Unkown";
}

void dump_ggml_tensor(const char* prefix, struct ggml_tensor* tensor, bool more = false)
{
    char output_buffer[1024];

    check_avoid(tensor);

    size_t len = 0;
    if (tensor->name) {
        len += snprintf(output_buffer + len, sizeof(output_buffer) - len, "%s%s: %s, [%ld, %ld, %ld, %ld], %s", 
            prefix, tensor->name, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            backend_name(tensor->backend));
    } else {
        len += snprintf(output_buffer + len, sizeof(output_buffer) - len, "%s%s: %s, [%ld, %ld, %ld, %ld], %s", 
            prefix, "none", ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            backend_name(tensor->backend));
    }

#if 0 // For debug
    if (tensor->data == NULL) {
        len += snprintf(output_buffer + len, sizeof(output_buffer) - len, ", data == NULL");
    } else {
        len += snprintf(output_buffer + len, sizeof(output_buffer) - len, ", data != NULL");
    }

    if (tensor->buffer == NULL) {
        len += snprintf(output_buffer + len, sizeof(output_buffer) - len, ", buffer == NULL");
    } else {
        len += snprintf(output_buffer + len, sizeof(output_buffer) - len, ", buffer != NULL");
    }
#endif

    // Set_device(0) for more -- on cpu is OK, on cuda is NO !
    if (more) {
        double avg;
        float min, max, value;

        min = FLT_MAX;
        max = FLT_MIN;
        avg = 0.0;
        for (size_t i = 0; i < (size_t)ggml_nelements(tensor); i++) {
            value = ggml_get_f32_1d(tensor, i);
            min = MIN(min, value);
            max = MAX(max, value);
            avg += value;
        }
        if (ggml_nelements(tensor) > 0) {
            avg /= ggml_nelements(tensor);
        }

        len += snprintf(output_buffer + len, sizeof(output_buffer) - len, ", min: %.6f, max: %.6f, mean: %.6f", min,
            max, (float)avg);
    }

    syslog_info("%s", output_buffer);
}

static float get_f32_from_buffer(void* data, ggml_type dtype, size_t i)
{
    void* base_data = (char*)data + i * ggml_type_size(dtype);

    switch (dtype) {
    case GGML_TYPE_I8: {
        return ((int8_t*)base_data)[0];
    } break;
    case GGML_TYPE_I16: {
        return ((int16_t*)base_data)[0];
    } break;
    case GGML_TYPE_I32: {
        return ((int32_t*)base_data)[0];
    } break;
    case GGML_TYPE_F16: {
        return ggml_fp16_to_fp32(((ggml_fp16_t*)base_data)[0]);
    } break;
    case GGML_TYPE_F32: {
        return ((float*)base_data)[0];
    } break;
    default: {
        GGML_ASSERT(false); // Invalid data type.
    } break;
    }

    return 0.0f;
}

static bool set_f32_to_buffer(void* data, ggml_type dtype, size_t i, float value)
{
    void* base_data = (char*)data + i * ggml_type_size(dtype);

    switch (dtype) {
    case GGML_TYPE_I8: {
        ((int8_t*)(base_data))[0] = (int8_t)value;
    } break;
    case GGML_TYPE_I16: {
        ((int16_t*)(base_data))[0] = (int16_t)value;
    } break;
    case GGML_TYPE_I32: {
        ((int32_t*)(base_data))[0] = (int32_t)value;
    } break;
    case GGML_TYPE_F16: {
        ((ggml_fp16_t*)(base_data))[0] = ggml_fp32_to_fp16(value);
    } break;
    case GGML_TYPE_F32: {
        ((float*)(base_data))[0] = value;
    } break;
    default: {
        GGML_ASSERT(false); // Invalid data type
        return false;
    } break;
    }

    return true;
}


static bool data_type_valid(ggml_type dtype)
{
    bool ok = (dtype == GGML_TYPE_I8 || dtype == GGML_TYPE_I16 || dtype == GGML_TYPE_I32 || dtype == GGML_TYPE_F16
        || dtype == GGML_TYPE_F32);
    if (!ok) {
        syslog_error("Tensor data is nor I8, I16, I32, F16 or F32");
    }
    return ok;
}

static bool same_data_shape(struct ggml_tensor* tensor, TENSOR* nt)
{
    // B, C, H, W
    bool ok = (nt->batch == (int)tensor->ne[3] && nt->chan == (int)tensor->ne[2] && nt->height == (int)tensor->ne[1]
        && nt->width == (int)tensor->ne[0]);

    if (!ok) {
        syslog_error("Tensor shape mismatch: expect[%d, %d, %d, %d], got [%d, %d, %d, %d]", (int)tensor->ne[3],
            (int)tensor->ne[2], (int)tensor->ne[1], (int)tensor->ne[0], nt->batch, nt->chan, nt->height, nt->width);
        return false;
    }

    return ok;
}

TENSOR* get_tensor_value(struct ggml_tensor* tensor, bool from_backend = false)
{
    CHECK_POINT(tensor);
    CHECK_POINT(data_type_valid(tensor->type));

    // B, C, H, W
    TENSOR* nt = tensor_create((int)tensor->ne[3], (int)tensor->ne[2], (int)tensor->ne[1], (int)tensor->ne[0]);
    CHECK_TENSOR(nt);

    size_t n = nt->batch * nt->chan * nt->height * nt->width;

    if (from_backend) {
        void* source_data = (void*)malloc(ggml_nbytes(tensor));
        if (! source_data) {
            tensor_destroy(nt);
        }
        CHECK_POINT(source_data);

        ggml_backend_tensor_get(tensor, source_data, 0, ggml_nbytes(tensor));

        for (size_t i = 0; i < n; i++)
            nt->data[i] = get_f32_from_buffer(source_data, tensor->type, i);

        free(source_data);

        return nt;
    }

    // Case from_backend == false
    for (size_t i = 0; i < n; i++)
        nt->data[i] = get_f32_from_buffer(tensor->data, tensor->type, i);

    return nt;
}

bool set_tensor_value(struct ggml_tensor* tensor, TENSOR* nt, bool to_backend = false)
{
    check_point(tensor);
    check_tensor(nt);

    // B, C, H, W
    check_point(same_data_shape(tensor, nt));
    check_point(data_type_valid(tensor->type));

    size_t n = nt->batch * nt->chan * nt->height * nt->width;
    if (to_backend) {
        void* destion_data = (void*)malloc(ggml_nbytes(tensor));
        check_point(destion_data);

        for (size_t i = 0; i < n; i++)
            set_f32_to_buffer(destion_data, tensor->type, i, nt->data[i]);

        ggml_backend_tensor_set(tensor, destion_data, 0, ggml_nbytes(tensor));
        free(destion_data);
        return true;
    }

    // Case to_backend == false
    for (size_t i = 0; i < n; i++)
        set_f32_to_buffer(tensor->data, tensor->type, i, nt->data[i]);

    return true;
}

static struct ggml_context* create_context(size_t mem_size = 10 * 1024 * 1024, bool static_alloc = true)
{
    struct ggml_init_params params = {
        /*.mem_size   =*/mem_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/static_alloc, // the tensors will be allocated later or not
    };

    struct ggml_context* ctx = ggml_init(params);
    CHECK_POINT(ctx);

    return ctx;
}
