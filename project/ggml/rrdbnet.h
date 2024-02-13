#ifndef __RRDBNET__H__
#define __RRDBNET__H__

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
    struct ggml_tensor* conv1_weight;  // torch.float32, [32, 64, 3, 3] 
    struct ggml_tensor* conv1_bias;  // torch.float32, [32] 
    struct ggml_tensor* conv2_weight;  // torch.float32, [32, 96, 3, 3] 
    struct ggml_tensor* conv2_bias;  // torch.float32, [32] 
    struct ggml_tensor* conv3_weight;  // torch.float32, [32, 128, 3, 3] 
    struct ggml_tensor* conv3_bias;  // torch.float32, [32] 
    struct ggml_tensor* conv4_weight;  // torch.float32, [32, 160, 3, 3] 
    struct ggml_tensor* conv4_bias;  // torch.float32, [32] 
    struct ggml_tensor* conv5_weight;  // torch.float32, [64, 192, 3, 3] 
    struct ggml_tensor* conv5_bias;  // torch.float32, [64]


    void create_weight_tensors(struct ggml_context* ctx) {
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

    void setup_weight_names(char *prefix) {
        set_tensor_name(conv1_weight, prefix, "conv1.weight");
        set_tensor_name(conv1_bias, prefix, "conv1.bias");
        set_tensor_name(conv2_weight, prefix, "conv2.weight");
        set_tensor_name(conv2_bias, prefix, "conv2.bias");
        set_tensor_name(conv3_weight, prefix, "conv3.weight");
        set_tensor_name(conv3_bias, prefix, "conv3.bias");
        set_tensor_name(conv4_weight, prefix, "conv4.weight");
        set_tensor_name(conv4_bias, prefix, "conv4.bias");
        set_tensor_name(conv5_weight, prefix, "conv5.weight");
        set_tensor_name(conv5_bias, prefix, "conv5.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        ggml_tensor* x1 = ggml_nn_conv_2d(ctx, x, conv1_weight, conv1_bias, 1, 1, 1, 1);
        x1 = ggml_leaky_relu(ctx, x1, 0.2f, true);

        // x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        ggml_tensor* x_cat = ggml_concat(ctx, x, x1);
        ggml_tensor* x2    = ggml_nn_conv_2d(ctx, x_cat, conv2_weight, conv2_bias, 1, 1, 1, 1);
        x2 = ggml_leaky_relu(ctx, x2, 0.2f, true);

        // x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x_cat = ggml_concat(ctx, x_cat, x2);
        ggml_tensor* x3 = ggml_nn_conv_2d(ctx, x_cat, conv3_weight, conv3_bias, 1, 1, 1, 1);
        x3 = ggml_leaky_relu(ctx, x3, 0.2f, true);

        // x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x_cat = ggml_concat(ctx, x_cat, x3);
        ggml_tensor* x4 = ggml_nn_conv_2d(ctx, x_cat, conv4_weight, conv4_bias, 1, 1, 1, 1);
        x4 = ggml_leaky_relu(ctx, x4, 0.2f, true);

        // self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x_cat = ggml_concat(ctx, x_cat, x4);
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


    void create_weight_tensors(struct ggml_context* ctx) {
        rdb1.create_weight_tensors(ctx);
        rdb2.create_weight_tensors(ctx);
        rdb3.create_weight_tensors(ctx);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "rdb1.");
        rdb1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "rdb2.");
        rdb2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "rdb3.");
        rdb3.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        ggml_tensor* out = x;
        out = rdb1.forward(ctx, out);
        out = rdb2.forward(ctx, out);
        out = rdb3.forward(ctx, out);

        // return out * 0.2 + x
        out = ggml_add(ctx, ggml_scale(ctx, out, 0.2), x);

        return out;

        // out = self.rdb1(x)
        // out = self.rdb2(out)
        // out = self.rdb3(out)
        // # Empirically, we use 0.2 to scale the residual for better performance
        // return out * 0.2 + x

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

struct RRDBNet : GGMLNetwork {
    // network hparams
    int MAX_H = 1024;
    int MAX_W = 1024;
    int MAX_TIMES = 4;
    int num_block = 6;
    int scale = 4;

    // network params
    struct ggml_tensor* conv_first_weight;  // torch.float32, [64, 3, 3, 3] 
    struct ggml_tensor* conv_first_bias;  // torch.float32, [64] 
    struct RRDB body_0;
    struct RRDB body_1;
    struct RRDB body_2;
    struct RRDB body_3;
    struct RRDB body_4;
    struct RRDB body_5;
    struct ggml_tensor* conv_body_weight;  // torch.float32, [64, 64, 3, 3] 
    struct ggml_tensor* conv_body_bias;  // torch.float32, [64] 
    struct ggml_tensor* conv_up1_weight;  // torch.float32, [64, 64, 3, 3] 
    struct ggml_tensor* conv_up1_bias;  // torch.float32, [64] 
    struct ggml_tensor* conv_up2_weight;  // torch.float32, [64, 64, 3, 3] 
    struct ggml_tensor* conv_up2_bias;  // torch.float32, [64] 
    struct ggml_tensor* conv_hr_weight;  // torch.float32, [64, 64, 3, 3] 
    struct ggml_tensor* conv_hr_bias;  // torch.float32, [64] 
    struct ggml_tensor* conv_last_weight;  // torch.float32, [3, 64, 3, 3] 
    struct ggml_tensor* conv_last_bias;  // torch.float32, [3]


    void create_weight_tensors(struct ggml_context* ctx) {
        conv_first_weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 3, 64);
        conv_first_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        body_0.create_weight_tensors(ctx);
        body_1.create_weight_tensors(ctx);
        body_2.create_weight_tensors(ctx);
        body_3.create_weight_tensors(ctx);
        body_4.create_weight_tensors(ctx);
        body_5.create_weight_tensors(ctx);
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

    void setup_weight_names(char *prefix) {
        char s[512];
        set_tensor_name(conv_first_weight, prefix, "conv_first.weight");
        set_tensor_name(conv_first_bias, prefix, "conv_first.bias");
        snprintf(s, sizeof(s), "%s%s", prefix, "body.0.");

        body_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "body.1.");
        body_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "body.2.");
        body_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "body.3.");
        body_3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "body.4.");
        body_4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "body.5.");
        body_5.setup_weight_names(s);
        set_tensor_name(conv_body_weight, prefix, "conv_body.weight");
        set_tensor_name(conv_body_bias, prefix, "conv_body.bias");
        set_tensor_name(conv_up1_weight, prefix, "conv_up1.weight");
        set_tensor_name(conv_up1_bias, prefix, "conv_up1.bias");
        set_tensor_name(conv_up2_weight, prefix, "conv_up2.weight");
        set_tensor_name(conv_up2_bias, prefix, "conv_up2.bias");
        set_tensor_name(conv_hr_weight, prefix, "conv_hr.weight");
        set_tensor_name(conv_hr_bias, prefix, "conv_hr.bias");
        set_tensor_name(conv_last_weight, prefix, "conv_last.weight");
        set_tensor_name(conv_last_bias, prefix, "conv_last.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, int eng_argc, struct ggml_tensor* eng_argv[]) {
        struct ggml_tensor* x = eng_argv[0];

        auto h = ggml_nn_conv_2d(ctx, x, conv_first_weight, conv_first_bias, 1, 1, 1, 1);

        auto body_h = h;

        // self.body(feat)
        body_h = body_0.forward(ctx, body_h);
        body_h = body_1.forward(ctx, body_h);
        body_h = body_2.forward(ctx, body_h);
        body_h = body_3.forward(ctx, body_h);
        body_h = body_4.forward(ctx, body_h);
        body_h = body_5.forward(ctx, body_h);

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

#endif // __RRDBNET__H
