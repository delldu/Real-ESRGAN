#ifndef __RRDBNET__H__
#define __RRDBNET__H__

#define GGML_ENGINE_IMPLEMENTATION
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

/*
 ResidualDenseBlock(
  (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
) */

struct ResidualDenseBlock {
    // network hparams
    // int num_features = 64;
    // int num_grow_ch = 32;

    // network params
    struct ggml_tensor* conv1_weight; // torch.float32, [32, 64, 3, 3]
    struct ggml_tensor* conv1_bias; // torch.float32, [32]
    struct ggml_tensor* conv2_weight; // torch.float32, [32, 96, 3, 3]
    struct ggml_tensor* conv2_bias; // torch.float32, [32]
    struct ggml_tensor* conv3_weight; // torch.float32, [32, 128, 3, 3]
    struct ggml_tensor* conv3_bias; // torch.float32, [32]
    struct ggml_tensor* conv4_weight; // torch.float32, [32, 160, 3, 3]
    struct ggml_tensor* conv4_bias; // torch.float32, [32]
    struct ggml_tensor* conv5_weight; // torch.float32, [64, 192, 3, 3]
    struct ggml_tensor* conv5_bias; // torch.float32, [64]

    void create_weight_tensors(struct ggml_context* ctx)
    {
        conv1_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 32);
        conv1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        conv2_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 96, 32);
        conv2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        conv3_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 128, 32);
        conv3_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        conv4_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 160, 32);
        conv4_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
        conv5_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 192, 64);
        conv5_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(conv1_weight, "%s%s", prefix, "conv1.weight");
        ggml_format_name(conv1_bias, "%s%s", prefix, "conv1.bias");
        ggml_format_name(conv2_weight, "%s%s", prefix, "conv2.weight");
        ggml_format_name(conv2_bias, "%s%s", prefix, "conv2.bias");
        ggml_format_name(conv3_weight, "%s%s", prefix, "conv3.weight");
        ggml_format_name(conv3_bias, "%s%s", prefix, "conv3.bias");
        ggml_format_name(conv4_weight, "%s%s", prefix, "conv4.weight");
        ggml_format_name(conv4_bias, "%s%s", prefix, "conv4.bias");
        ggml_format_name(conv5_weight, "%s%s", prefix, "conv5.weight");
        ggml_format_name(conv5_bias, "%s%s", prefix, "conv5.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        ggml_tensor* x1 = ggml_nn_conv_2d(ctx, x, conv1_weight, conv1_bias, 1, 1, 1, 1);
        x1 = ggml_leaky_relu(ctx, x1, 0.2f, true);

        // x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        ggml_tensor* x_cat = ggml_concat(ctx, x, x1, 2);
        ggml_tensor* x2 = ggml_nn_conv_2d(ctx, x_cat, conv2_weight, conv2_bias, 1, 1, 1, 1);
        x2 = ggml_leaky_relu(ctx, x2, 0.2f, true);

        // x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x_cat = ggml_concat(ctx, x_cat, x2, 2);
        ggml_tensor* x3 = ggml_nn_conv_2d(ctx, x_cat, conv3_weight, conv3_bias, 1, 1, 1, 1);
        x3 = ggml_leaky_relu(ctx, x3, 0.2f, true);

        // x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x_cat = ggml_concat(ctx, x_cat, x3, 2);
        ggml_tensor* x4 = ggml_nn_conv_2d(ctx, x_cat, conv4_weight, conv4_bias, 1, 1, 1, 1);
        x4 = ggml_leaky_relu(ctx, x4, 0.2f, true);

        // self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x_cat = ggml_concat(ctx, x_cat, x4, 2);
        ggml_tensor* x5 = ggml_nn_conv_2d(ctx, x_cat, conv5_weight, conv5_bias, 1, 1, 1, 1);

        // return x5 * 0.2 + x
        x5 = ggml_add(ctx, ggml_scale(ctx, x5, 0.2), x);
        return x5;
    }
};

/*
 RRDB(
  (rdb1): ResidualDenseBlock(
    (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
  )
  (rdb2): ResidualDenseBlock(
    (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
  )
  (rdb3): ResidualDenseBlock(
    (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
  )
) */

struct RRDB {
    // network hparams

    // network params
    struct ResidualDenseBlock rdb1;
    struct ResidualDenseBlock rdb2;
    struct ResidualDenseBlock rdb3;

    void create_weight_tensors(struct ggml_context* ctx)
    {
        rdb1.create_weight_tensors(ctx);
        rdb2.create_weight_tensors(ctx);
        rdb3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "rdb1.");
        rdb1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "rdb2.");
        rdb2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "rdb3.");
        rdb3.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        ggml_tensor* out = x;
        out = rdb1.forward(ctx, out);
        out = rdb2.forward(ctx, out);
        out = rdb3.forward(ctx, out);

        // return out * 0.2 + x
        out = ggml_add(ctx, ggml_scale(ctx, out, 0.2), x);

        return out;
    }
};

/*
 RRDBNet(
  (conv_first): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (body): Sequential(
    (0): RRDB(
      (rdb1): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb2): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb3): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
    )
    (1): RRDB(
      (rdb1): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb2): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb3): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
    )
    (2): RRDB(
      (rdb1): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb2): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb3): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
    )
    (3): RRDB(
      (rdb1): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb2): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb3): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
    )
    (4): RRDB(
      (rdb1): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb2): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb3): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
    )
    (5): RRDB(
      (rdb1): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb2): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (rdb3): ResidualDenseBlock(
        (conv1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv3): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv4): Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv5): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
      )
    )
  )
  (conv_body): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_up1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_up2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_hr): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_last): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (lrelu): LeakyReLU(negative_slope=0.2, inplace=True)
) */

struct Anime4x : GGMLNetwork {
    // network hparams
    const int MAX_H = 512;
    const int MAX_W = 512;
    const int MAX_TIMES = 4;
    // const int num_block = 6;
    const int scale = 4;

    // network params
    struct ggml_tensor* conv_first_weight; // torch.float32, [64, 3, 3, 3]
    struct ggml_tensor* conv_first_bias; // torch.float32, [64]
    struct RRDB body_[6];
    struct ggml_tensor* conv_body_weight; // torch.float32, [64, 64, 3, 3]
    struct ggml_tensor* conv_body_bias; // torch.float32, [64]
    struct ggml_tensor* conv_up1_weight; // torch.float32, [64, 64, 3, 3]
    struct ggml_tensor* conv_up1_bias; // torch.float32, [64]
    struct ggml_tensor* conv_up2_weight; // torch.float32, [64, 64, 3, 3]
    struct ggml_tensor* conv_up2_bias; // torch.float32, [64]
    struct ggml_tensor* conv_hr_weight; // torch.float32, [64, 64, 3, 3]
    struct ggml_tensor* conv_hr_bias; // torch.float32, [64]
    struct ggml_tensor* conv_last_weight; // torch.float32, [3, 64, 3, 3]
    struct ggml_tensor* conv_last_bias; // torch.float32, [3]

    void create_weight_tensors(struct ggml_context* ctx)
    {
        conv_first_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 3, 64);
        conv_first_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        for (int i = 0; i < ARRAY_SIZE(body_); i++)
            body_[i].create_weight_tensors(ctx);
        conv_body_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 64);
        conv_body_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv_up1_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 64);
        conv_up1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv_up2_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 64);
        conv_up2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv_hr_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 64);
        conv_hr_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv_last_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 3);
        conv_last_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[512];
        ggml_format_name(conv_first_weight, "%s%s", prefix, "conv_first.weight");
        ggml_format_name(conv_first_bias, "%s%s", prefix, "conv_first.bias");

        for (int i = 0; i < ARRAY_SIZE(body_); i++) {
            // snprintf(s, sizeof(s), "%s%s", prefix, "body.0.");
            snprintf(s, sizeof(s), "%sbody.%d.", prefix, i);
            body_[i].setup_weight_names(s);
        }
        ggml_format_name(conv_body_weight, "%s%s", prefix, "conv_body.weight");
        ggml_format_name(conv_body_bias, "%s%s", prefix, "conv_body.bias");
        ggml_format_name(conv_up1_weight, "%s%s", prefix, "conv_up1.weight");
        ggml_format_name(conv_up1_bias, "%s%s", prefix, "conv_up1.bias");
        ggml_format_name(conv_up2_weight, "%s%s", prefix, "conv_up2.weight");
        ggml_format_name(conv_up2_bias, "%s%s", prefix, "conv_up2.bias");
        ggml_format_name(conv_hr_weight, "%s%s", prefix, "conv_hr.weight");
        ggml_format_name(conv_hr_bias, "%s%s", prefix, "conv_hr.bias");
        ggml_format_name(conv_last_weight, "%s%s", prefix, "conv_last.weight");
        ggml_format_name(conv_last_bias, "%s%s", prefix, "conv_last.bias");
    }

    // size_t get_graph_size()
    // {
    //     return GGML_DEFAULT_GRAPH_SIZE * 2; // 2048 ==> 4096
    // }

    struct ggml_tensor* forward(struct ggml_context* ctx, int eng_argc, struct ggml_tensor* eng_argv[])
    {
        struct ggml_tensor* x = eng_argv[0];

        auto h = ggml_nn_conv_2d(ctx, x, conv_first_weight, conv_first_bias, 1, 1, 1, 1);

        auto body_h = h;

        // self.body(feat)
        for (int i = 0; i < ARRAY_SIZE(body_); i++)
            body_h = body_[i].forward(ctx, body_h);

        // body_feat = self.conv_body(self.body(feat))
        body_h = ggml_nn_conv_2d(ctx, body_h, conv_body_weight, conv_body_bias, 1, 1, 1, 1);

        // feat = feat + body_feat
        h = ggml_add(ctx, h, body_h);

        // upsample
        // feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        h = ggml_upscale(ctx, h, 2);
        h = ggml_nn_conv_2d(ctx, h, conv_up1_weight, conv_up1_bias, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx, h, 0.2f, true);

        // feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        h = ggml_upscale(ctx, h, 2);
        h = ggml_nn_conv_2d(ctx, h, conv_up2_weight, conv_up2_bias, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx, h, 0.2f, true);

        // out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        h = ggml_nn_conv_2d(ctx, h, conv_hr_weight, conv_hr_bias, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx, h, 0.2f, true);

        h = ggml_nn_conv_2d(ctx, h, conv_last_weight, conv_last_bias, 1, 1, 1, 1);

        return ggml_clamp(ctx, h, 0.0, 1.0);
    }
};

int anime4x_predict(int device, int n, char *input_files[], char *output_dir)
{
    char *p, output_fname[512];
    TENSOR *input_tensor, *output_tensor, *tensor_argv[1] = {};

    if (n < 1)
        return RET_OK;

    make_dir(output_dir);

    struct Anime4x net;
    net.set_device(device); // 0 -- cpu, 1 -- cuda
    net.load("models/anime4x.gguf", "");
    net.start_engine();
    // net.dump();

    for (int i = 0; i < n; i++) {
        p = strrchr(input_files[i], '/');
        p = (!p) ? input_files[i] : p + 1;
        snprintf(output_fname, sizeof(output_fname) - 1, "%s/%s", output_dir, p);

        syslog_info("Anime4x predict %s to %s ...", input_files[i], output_fname);

        input_tensor = tensor_load_image(input_files[i], 0 /*without alpha*/);
        tensor_resizepad_(input_tensor, net.MAX_H, net.MAX_W, net.MAX_TIMES);
        check_tensor(input_tensor);

        tensor_argv[0] = input_tensor;
        output_tensor = net.engine_forward(ARRAY_SIZE(tensor_argv), tensor_argv);
        check_tensor(output_tensor);
        if (output_tensor) {
            tensor_saveas_image(output_tensor, 0 /*batch 0*/, output_fname);
            tensor_destroy(output_tensor);
        }

        tensor_destroy(input_tensor);
    }

    // net.dump();

    net.stop_engine();

    return RET_OK;
}

struct Zoom4x : GGMLNetwork {
    // network hparams
    const int MAX_H = 512;
    const int MAX_W = 512;
    const int MAX_TIMES = 4;
    // const int num_block = 23;
    const int scale = 4;

    // network params
    struct ggml_tensor* conv_first_weight; // torch.float32, [64, 3, 3, 3]
    struct ggml_tensor* conv_first_bias; // torch.float32, [64]
    struct RRDB body_[23];
    struct ggml_tensor* conv_body_weight; // torch.float32, [64, 64, 3, 3]
    struct ggml_tensor* conv_body_bias; // torch.float32, [64]
    struct ggml_tensor* conv_up1_weight; // torch.float32, [64, 64, 3, 3]
    struct ggml_tensor* conv_up1_bias; // torch.float32, [64]
    struct ggml_tensor* conv_up2_weight; // torch.float32, [64, 64, 3, 3]
    struct ggml_tensor* conv_up2_bias; // torch.float32, [64]
    struct ggml_tensor* conv_hr_weight; // torch.float32, [64, 64, 3, 3]
    struct ggml_tensor* conv_hr_bias; // torch.float32, [64]
    struct ggml_tensor* conv_last_weight; // torch.float32, [3, 64, 3, 3]
    struct ggml_tensor* conv_last_bias; // torch.float32, [3]

    void create_weight_tensors(struct ggml_context* ctx)
    {
        conv_first_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 3, 64);
        conv_first_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        for (int i = 0; i < ARRAY_SIZE(body_); i++)
            body_[i].create_weight_tensors(ctx);
        conv_body_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 64);
        conv_body_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv_up1_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 64);
        conv_up1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv_up2_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 64);
        conv_up2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv_hr_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 64);
        conv_hr_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        conv_last_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 64, 3);
        conv_last_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    }


    void setup_weight_names(const char* prefix)
    {
        char s[512];
        ggml_format_name(conv_first_weight, "%s%s", prefix, "conv_first.weight");
        ggml_format_name(conv_first_bias, "%s%s", prefix, "conv_first.bias");

        for (int i = 0; i < ARRAY_SIZE(body_); i++) {
            // snprintf(s, sizeof(s), "%s%s", prefix, "body.0.");
            snprintf(s, sizeof(s), "%sbody.%d.", prefix, i);
            body_[i].setup_weight_names(s);
        }

        ggml_format_name(conv_body_weight, "%s%s", prefix, "conv_body.weight");
        ggml_format_name(conv_body_bias, "%s%s", prefix, "conv_body.bias");
        ggml_format_name(conv_up1_weight, "%s%s", prefix, "conv_up1.weight");
        ggml_format_name(conv_up1_bias, "%s%s", prefix, "conv_up1.bias");
        ggml_format_name(conv_up2_weight, "%s%s", prefix, "conv_up2.weight");
        ggml_format_name(conv_up2_bias, "%s%s", prefix, "conv_up2.bias");
        ggml_format_name(conv_hr_weight, "%s%s", prefix, "conv_hr.weight");
        ggml_format_name(conv_hr_bias, "%s%s", prefix, "conv_hr.bias");
        ggml_format_name(conv_last_weight, "%s%s", prefix, "conv_last.weight");
        ggml_format_name(conv_last_bias, "%s%s", prefix, "conv_last.bias");
    }

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 2; // 2048 ==> 4096
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, int eng_argc, struct ggml_tensor* eng_argv[])
    {
        struct ggml_tensor* x = eng_argv[0];

        auto h = ggml_nn_conv_2d(ctx, x, conv_first_weight, conv_first_bias, 1, 1, 1, 1);

        auto body_h = h;

        // self.body(feat)
        for (int i = 0; i < ARRAY_SIZE(body_); i++)
            body_h = body_[i].forward(ctx, body_h);

        // body_feat = self.conv_body(self.body(feat))
        body_h = ggml_nn_conv_2d(ctx, body_h, conv_body_weight, conv_body_bias, 1, 1, 1, 1);

        // feat = feat + body_feat
        h = ggml_add(ctx, h, body_h);

        // upsample
        // feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        h = ggml_upscale(ctx, h, 2);
        h = ggml_nn_conv_2d(ctx, h, conv_up1_weight, conv_up1_bias, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx, h, 0.2f, true);

        // feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        h = ggml_upscale(ctx, h, 2);
        h = ggml_nn_conv_2d(ctx, h, conv_up2_weight, conv_up2_bias, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx, h, 0.2f, true);

        // out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        h = ggml_nn_conv_2d(ctx, h, conv_hr_weight, conv_hr_bias, 1, 1, 1, 1);
        h = ggml_leaky_relu(ctx, h, 0.2f, true);

        h = ggml_nn_conv_2d(ctx, h, conv_last_weight, conv_last_bias, 1, 1, 1, 1);

        return ggml_clamp(ctx, h, 0.0, 1.0);
    }
};


int zoom4x_predict(int device, int n, char *input_files[], char *output_dir)
{
    char *p, output_fname[512];
    TENSOR *input_tensor, *output_tensor, *tensor_argv[1] = {};

    if (n < 1)
        return RET_OK;

    make_dir(output_dir);

    struct Zoom4x net;
    net.set_device(device); // 0 -- cpu, 1 -- cuda
    net.load("models/zoom4x.gguf", "");
    net.start_engine();
    // net.dump();

    for (int i = 0; i < n; i++) {
        p = strrchr(input_files[i], '/');
        p = (!p) ? input_files[i] : p + 1;
        snprintf(output_fname, sizeof(output_fname) - 1, "%s/%s", output_dir, p);

        syslog_info("Zoom4x predict %s to %s ...", input_files[i], output_fname);

        input_tensor = tensor_load_image(input_files[i], 0 /*without alpha*/);
        tensor_resizepad_(input_tensor, net.MAX_H, net.MAX_W, net.MAX_TIMES);

        check_tensor(input_tensor);

        tensor_argv[0] = input_tensor;
        output_tensor = net.engine_forward(ARRAY_SIZE(tensor_argv), tensor_argv);
        check_tensor(output_tensor);
        if (output_tensor) {
            tensor_saveas_image(output_tensor, 0 /*batch 0*/, output_fname);
            tensor_destroy(output_tensor);
        }

        tensor_destroy(input_tensor);
    }

    net.stop_engine();

    return RET_OK;
}


#endif // __RRDBNET__H
