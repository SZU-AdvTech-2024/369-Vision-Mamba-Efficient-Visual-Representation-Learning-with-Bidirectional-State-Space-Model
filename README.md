# 使用说明
1. python环境安装以及Mamba相关的causal_conv1d和mamba-1p1p1的安装请参考：https://github.com/hustvl/Vim
2. 本代码应用于SUN-SEG和CVC-Clinic DB两个公开的息肉数据集
3. 配置文件于scripts/config.py, 根据需求进行相应的配置
4. 训练：python my_train.py, 训练过程中会将最优模型保存于\experiments\xxx\ckpt_epoch_x.pth
5. 测试：python my_test.py, 测试前需在__main__中填写测试数据集相关及ckpt路径


# tree
[01;34m.[0m
├── [01;34meval[0m
│   ├── dice_score.py
│   ├── eval.sh
│   ├── evaluator.py
│   ├── metrics.py
│   ├── [01;34m__pycache__[0m
│   │   ├── dice_score.cpython-310.pyc
│   │   ├── evaluator.cpython-310.pyc
│   │   ├── evaluator.cpython-36.pyc
│   │   ├── metrics.cpython-310.pyc
│   │   └── metrics.cpython-36.pyc
│   ├── README.md
│   └── vps_evaluator.py
├── [01;34mlib[0m
│   ├── [01;34mbackbone[0m
│   │   ├── CompressEncoder.py
│   │   ├── Decoder.py
│   │   ├── LightRFB.py
│   │   ├── pvt_v2.py
│   │   ├── [01;34m__pycache__[0m
│   │   │   ├── CompressEncoder.cpython-310.pyc
│   │   │   ├── Decoder.cpython-310.pyc
│   │   │   ├── LightRFB.cpython-310.pyc
│   │   │   ├── pvt_v2.cpython-310.pyc
│   │   │   └── Res2Net_v1b.cpython-310.pyc
│   │   └── Res2Net_v1b.py
│   ├── [01;34mdataloader[0m
│   │   ├── dataloader.py
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   │   ├── [01;34m__pycache__[0m
│   │   │   ├── dataloader.cpython-310.pyc
│   │   │   ├── dataloader.cpython-36.pyc
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   ├── preprocess.cpython-310.pyc
│   │   │   └── preprocess.cpython-36.pyc
│   │   └── statistics.pth
│   ├── [01;34mg_cascade[0m
│   │   ├── decoders.py
│   │   ├── [01;34mgcn_lib[0m
│   │   │   ├── __init__.py
│   │   │   ├── pos_embed.py
│   │   │   ├── [01;34m__pycache__[0m
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   ├── pos_embed.cpython-310.pyc
│   │   │   │   ├── torch_edge.cpython-310.pyc
│   │   │   │   ├── torch_nn.cpython-310.pyc
│   │   │   │   └── torch_vertex.cpython-310.pyc
│   │   │   ├── torch_edge.py
│   │   │   ├── torch_nn.py
│   │   │   └── torch_vertex.py
│   │   ├── maxxvit_4out.py
│   │   ├── [01;34mmodels_timm[0m
│   │   │   ├── beit.py
│   │   │   ├── byoanet.py
│   │   │   ├── byobnet.py
│   │   │   ├── cait.py
│   │   │   ├── coat.py
│   │   │   ├── convit.py
│   │   │   ├── convmixer.py
│   │   │   ├── convnext.py
│   │   │   ├── crossvit.py
│   │   │   ├── cspnet.py
│   │   │   ├── deit.py
│   │   │   ├── densenet.py
│   │   │   ├── dla.py
│   │   │   ├── dpn.py
│   │   │   ├── edgenext.py
│   │   │   ├── efficientformer.py
│   │   │   ├── efficientnet_blocks.py
│   │   │   ├── efficientnet_builder.py
│   │   │   ├── efficientnet.py
│   │   │   ├── factory.py
│   │   │   ├── features.py
│   │   │   ├── fx_features.py
│   │   │   ├── gcvit.py
│   │   │   ├── ghostnet.py
│   │   │   ├── gluon_resnet.py
│   │   │   ├── gluon_xception.py
│   │   │   ├── hardcorenas.py
│   │   │   ├── helpers.py
│   │   │   ├── hrnet.py
│   │   │   ├── hub.py
│   │   │   ├── inception_resnet_v2.py
│   │   │   ├── inception_v3.py
│   │   │   ├── inception_v4.py
│   │   │   ├── __init__.py
│   │   │   ├── [01;34mlayers[0m
│   │   │   │   ├── activations_jit.py
│   │   │   │   ├── activations_me.py
│   │   │   │   ├── activations.py
│   │   │   │   ├── adaptive_avgmax_pool.py
│   │   │   │   ├── attention_pool2d.py
│   │   │   │   ├── blur_pool.py
│   │   │   │   ├── bottleneck_attn.py
│   │   │   │   ├── cbam.py
│   │   │   │   ├── classifier.py
│   │   │   │   ├── cond_conv2d.py
│   │   │   │   ├── config.py
│   │   │   │   ├── conv2d_same.py
│   │   │   │   ├── conv_bn_act.py
│   │   │   │   ├── create_act.py
│   │   │   │   ├── create_attn.py
│   │   │   │   ├── create_conv2d.py
│   │   │   │   ├── create_norm_act.py
│   │   │   │   ├── create_norm.py
│   │   │   │   ├── drop.py
│   │   │   │   ├── eca.py
│   │   │   │   ├── evo_norm.py
│   │   │   │   ├── fast_norm.py
│   │   │   │   ├── filter_response_norm.py
│   │   │   │   ├── gather_excite.py
│   │   │   │   ├── global_context.py
│   │   │   │   ├── halo_attn.py
│   │   │   │   ├── helpers.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── inplace_abn.py
│   │   │   │   ├── lambda_layer.py
│   │   │   │   ├── linear.py
│   │   │   │   ├── median_pool.py
│   │   │   │   ├── mixed_conv2d.py
│   │   │   │   ├── ml_decoder.py
│   │   │   │   ├── mlp.py
│   │   │   │   ├── non_local_attn.py
│   │   │   │   ├── norm_act.py
│   │   │   │   ├── norm.py
│   │   │   │   ├── padding.py
│   │   │   │   ├── patch_embed.py
│   │   │   │   ├── pool2d_same.py
│   │   │   │   ├── pos_embed.py
│   │   │   │   ├── [01;34m__pycache__[0m
│   │   │   │   │   ├── activations.cpython-310.pyc
│   │   │   │   │   ├── activations_jit.cpython-310.pyc
│   │   │   │   │   ├── activations_me.cpython-310.pyc
│   │   │   │   │   ├── adaptive_avgmax_pool.cpython-310.pyc
│   │   │   │   │   ├── blur_pool.cpython-310.pyc
│   │   │   │   │   ├── bottleneck_attn.cpython-310.pyc
│   │   │   │   │   ├── cbam.cpython-310.pyc
│   │   │   │   │   ├── classifier.cpython-310.pyc
│   │   │   │   │   ├── cond_conv2d.cpython-310.pyc
│   │   │   │   │   ├── config.cpython-310.pyc
│   │   │   │   │   ├── conv2d_same.cpython-310.pyc
│   │   │   │   │   ├── conv_bn_act.cpython-310.pyc
│   │   │   │   │   ├── create_act.cpython-310.pyc
│   │   │   │   │   ├── create_attn.cpython-310.pyc
│   │   │   │   │   ├── create_conv2d.cpython-310.pyc
│   │   │   │   │   ├── create_norm_act.cpython-310.pyc
│   │   │   │   │   ├── create_norm.cpython-310.pyc
│   │   │   │   │   ├── drop.cpython-310.pyc
│   │   │   │   │   ├── eca.cpython-310.pyc
│   │   │   │   │   ├── evo_norm.cpython-310.pyc
│   │   │   │   │   ├── fast_norm.cpython-310.pyc
│   │   │   │   │   ├── filter_response_norm.cpython-310.pyc
│   │   │   │   │   ├── gather_excite.cpython-310.pyc
│   │   │   │   │   ├── global_context.cpython-310.pyc
│   │   │   │   │   ├── halo_attn.cpython-310.pyc
│   │   │   │   │   ├── helpers.cpython-310.pyc
│   │   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   │   ├── inplace_abn.cpython-310.pyc
│   │   │   │   │   ├── lambda_layer.cpython-310.pyc
│   │   │   │   │   ├── linear.cpython-310.pyc
│   │   │   │   │   ├── mixed_conv2d.cpython-310.pyc
│   │   │   │   │   ├── mlp.cpython-310.pyc
│   │   │   │   │   ├── non_local_attn.cpython-310.pyc
│   │   │   │   │   ├── norm_act.cpython-310.pyc
│   │   │   │   │   ├── norm.cpython-310.pyc
│   │   │   │   │   ├── padding.cpython-310.pyc
│   │   │   │   │   ├── patch_embed.cpython-310.pyc
│   │   │   │   │   ├── pool2d_same.cpython-310.pyc
│   │   │   │   │   ├── selective_kernel.cpython-310.pyc
│   │   │   │   │   ├── separable_conv.cpython-310.pyc
│   │   │   │   │   ├── space_to_depth.cpython-310.pyc
│   │   │   │   │   ├── split_attn.cpython-310.pyc
│   │   │   │   │   ├── split_batchnorm.cpython-310.pyc
│   │   │   │   │   ├── squeeze_excite.cpython-310.pyc
│   │   │   │   │   ├── std_conv.cpython-310.pyc
│   │   │   │   │   ├── test_time_pool.cpython-310.pyc
│   │   │   │   │   ├── trace_utils.cpython-310.pyc
│   │   │   │   │   └── weight_init.cpython-310.pyc
│   │   │   │   ├── selective_kernel.py
│   │   │   │   ├── separable_conv.py
│   │   │   │   ├── space_to_depth.py
│   │   │   │   ├── split_attn.py
│   │   │   │   ├── split_batchnorm.py
│   │   │   │   ├── squeeze_excite.py
│   │   │   │   ├── std_conv.py
│   │   │   │   ├── test_time_pool.py
│   │   │   │   ├── trace_utils.py
│   │   │   │   └── weight_init.py
│   │   │   ├── levit.py
│   │   │   ├── maxxvit.py
│   │   │   ├── mlp_mixer.py
│   │   │   ├── mobilenetv3.py
│   │   │   ├── mobilevit.py
│   │   │   ├── mvitv2.py
│   │   │   ├── nasnet.py
│   │   │   ├── nest.py
│   │   │   ├── nfnet.py
│   │   │   ├── pit.py
│   │   │   ├── pnasnet.py
│   │   │   ├── poolformer.py
│   │   │   ├── [01;34mpruned[0m
│   │   │   │   ├── ecaresnet101d_pruned.txt
│   │   │   │   ├── ecaresnet50d_pruned.txt
│   │   │   │   ├── efficientnet_b1_pruned.txt
│   │   │   │   ├── efficientnet_b2_pruned.txt
│   │   │   │   └── efficientnet_b3_pruned.txt
│   │   │   ├── pvt_v2.py
│   │   │   ├── [01;34m__pycache__[0m
│   │   │   │   ├── beit.cpython-310.pyc
│   │   │   │   ├── features.cpython-310.pyc
│   │   │   │   ├── fx_features.cpython-310.pyc
│   │   │   │   ├── helpers.cpython-310.pyc
│   │   │   │   ├── hub.cpython-310.pyc
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   ├── registry.cpython-310.pyc
│   │   │   │   └── vision_transformer.cpython-310.pyc
│   │   │   ├── registry.py
│   │   │   ├── regnet.py
│   │   │   ├── res2net.py
│   │   │   ├── resnest.py
│   │   │   ├── resnet.py
│   │   │   ├── resnetv2.py
│   │   │   ├── rexnet.py
│   │   │   ├── selecsls.py
│   │   │   ├── senet.py
│   │   │   ├── sequencer.py
│   │   │   ├── sknet.py
│   │   │   ├── swin_transformer.py
│   │   │   ├── swin_transformer_v2_cr.py
│   │   │   ├── swin_transformer_v2.py
│   │   │   ├── tnt.py
│   │   │   ├── tresnet.py
│   │   │   ├── twins.py
│   │   │   ├── vgg.py
│   │   │   ├── visformer.py
│   │   │   ├── vision_transformer_hybrid.py
│   │   │   ├── vision_transformer.py
│   │   │   ├── vision_transformer_relpos.py
│   │   │   ├── volo.py
│   │   │   ├── vovnet.py
│   │   │   ├── xception_aligned.py
│   │   │   ├── xception.py
│   │   │   └── xcit.py
│   │   ├── networks.py
│   │   ├── pvtv2.py
│   │   ├── [01;34m__pycache__[0m
│   │   │   ├── decoders.cpython-310.pyc
│   │   │   ├── maxxvit_4out.cpython-310.pyc
│   │   │   ├── networks.cpython-310.pyc
│   │   │   ├── pvtv2.cpython-310.pyc
│   │   │   └── pyramid_vig.cpython-310.pyc
│   │   └── pyramid_vig.py
│   ├── __init__.py
│   ├── [01;34mmamba[0m
│   │   ├── =1.1.0
│   │   ├── [01;34mcausal_conv1d[0m
│   │   │   ├── =1.1.0
│   │   │   ├── AUTHORS
│   │   │   ├── [01;34mbuild[0m
│   │   │   │   ├── [01;34mlib.linux-x86_64-cpython-310[0m
│   │   │   │   │   └── [01;32mcausal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so[0m
│   │   │   │   └── [01;34mtemp.linux-x86_64-cpython-310[0m
│   │   │   │       ├── build.ninja
│   │   │   │       ├── [01;34mcsrc[0m
│   │   │   │       │   ├── causal_conv1d_bwd.o
│   │   │   │       │   ├── causal_conv1d_fwd.o
│   │   │   │       │   ├── causal_conv1d.o
│   │   │   │       │   └── causal_conv1d_update.o
│   │   │   │       ├── .ninja_deps
│   │   │   │       └── .ninja_log
│   │   │   ├── [01;34mcausal_conv1d[0m
│   │   │   │   ├── causal_conv1d_interface.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── [01;34m__pycache__[0m
│   │   │   │       ├── causal_conv1d_interface.cpython-310.pyc
│   │   │   │       └── __init__.cpython-310.pyc
│   │   │   ├── [01;32mcausal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so[0m
│   │   │   ├── [01;34mcausal_conv1d.egg-info[0m
│   │   │   │   ├── dependency_links.txt
│   │   │   │   ├── PKG-INFO
│   │   │   │   ├── requires.txt
│   │   │   │   ├── SOURCES.txt
│   │   │   │   └── top_level.txt
│   │   │   ├── [01;34mcsrc[0m
│   │   │   │   ├── causal_conv1d_bwd.cu
│   │   │   │   ├── causal_conv1d_common.h
│   │   │   │   ├── causal_conv1d.cpp
│   │   │   │   ├── causal_conv1d_fwd.cu
│   │   │   │   ├── causal_conv1d.h
│   │   │   │   ├── causal_conv1d_update.cu
│   │   │   │   └── static_switch.h
│   │   │   ├── LICENSE
│   │   │   ├── README.md
│   │   │   ├── setup.py
│   │   │   └── [01;34mtests[0m
│   │   │       └── test_causal_conv1d.py
│   │   ├── [01;34mmamba-1p1p1[0m
│   │   │   ├── [01;34massets[0m
│   │   │   │   └── [01;35mselection.png[0m
│   │   │   ├── AUTHORS
│   │   │   ├── [01;34mbenchmarks[0m
│   │   │   │   └── benchmark_generation_mamba_simple.py
│   │   │   ├── [01;34mbuild[0m
│   │   │   │   ├── [01;34mlib.linux-x86_64-cpython-310[0m
│   │   │   │   │   └── [01;32mselective_scan_cuda.cpython-310-x86_64-linux-gnu.so[0m
│   │   │   │   └── [01;34mtemp.linux-x86_64-cpython-310[0m
│   │   │   │       ├── build.ninja
│   │   │   │       ├── [01;34mcsrc[0m
│   │   │   │       │   └── [01;34mselective_scan[0m
│   │   │   │       │       ├── selective_scan_bwd_bf16_complex.o
│   │   │   │       │       ├── selective_scan_bwd_bf16_real.o
│   │   │   │       │       ├── selective_scan_bwd_fp16_complex.o
│   │   │   │       │       ├── selective_scan_bwd_fp16_real.o
│   │   │   │       │       ├── selective_scan_bwd_fp32_complex.o
│   │   │   │       │       ├── selective_scan_bwd_fp32_real.o
│   │   │   │       │       ├── selective_scan_fwd_bf16.o
│   │   │   │       │       ├── selective_scan_fwd_fp16.o
│   │   │   │       │       ├── selective_scan_fwd_fp32.o
│   │   │   │       │       └── selective_scan.o
│   │   │   │       ├── .ninja_deps
│   │   │   │       └── .ninja_log
│   │   │   ├── [01;34mcsrc[0m
│   │   │   │   └── [01;34mselective_scan[0m
│   │   │   │       ├── reverse_scan.cuh
│   │   │   │       ├── selective_scan_bwd_bf16_complex.cu
│   │   │   │       ├── selective_scan_bwd_bf16_real.cu
│   │   │   │       ├── selective_scan_bwd_fp16_complex.cu
│   │   │   │       ├── selective_scan_bwd_fp16_real.cu
│   │   │   │       ├── selective_scan_bwd_fp32_complex.cu
│   │   │   │       ├── selective_scan_bwd_fp32_real.cu
│   │   │   │       ├── selective_scan_bwd_kernel.cuh
│   │   │   │       ├── selective_scan_common.h
│   │   │   │       ├── selective_scan.cpp
│   │   │   │       ├── selective_scan_fwd_bf16.cu
│   │   │   │       ├── selective_scan_fwd_fp16.cu
│   │   │   │       ├── selective_scan_fwd_fp32.cu
│   │   │   │       ├── selective_scan_fwd_kernel.cuh
│   │   │   │       ├── selective_scan.h
│   │   │   │       ├── static_switch.h
│   │   │   │       └── uninitialized_copy.cuh
│   │   │   ├── [01;34mevals[0m
│   │   │   │   └── lm_harness_eval.py
│   │   │   ├── [01;34m.github[0m
│   │   │   │   └── [01;34mworkflows[0m
│   │   │   │       └── publish.yaml
│   │   │   ├── .gitignore
│   │   │   ├── .gitmodules
│   │   │   ├── LICENSE
│   │   │   ├── [01;34mmamba_ssm[0m
│   │   │   │   ├── __init__.py
│   │   │   │   ├── [01;34mmodels[0m
│   │   │   │   │   ├── config_mamba.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── mixer_seq_simple.py
│   │   │   │   │   └── [01;34m__pycache__[0m
│   │   │   │   │       ├── config_mamba.cpython-310.pyc
│   │   │   │   │       ├── __init__.cpython-310.pyc
│   │   │   │   │       └── mixer_seq_simple.cpython-310.pyc
│   │   │   │   ├── [01;34mmodules[0m
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── mamba_simple.py
│   │   │   │   │   └── [01;34m__pycache__[0m
│   │   │   │   │       ├── __init__.cpython-310.pyc
│   │   │   │   │       └── mamba_simple.cpython-310.pyc
│   │   │   │   ├── [01;34mops[0m
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── [01;34m__pycache__[0m
│   │   │   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   │   │   └── selective_scan_interface.cpython-310.pyc
│   │   │   │   │   ├── selective_scan_interface.py
│   │   │   │   │   └── [01;34mtriton[0m
│   │   │   │   │       ├── __init__.py
│   │   │   │   │       ├── layernorm.py
│   │   │   │   │       ├── [01;34m__pycache__[0m
│   │   │   │   │       │   ├── __init__.cpython-310.pyc
│   │   │   │   │       │   ├── layernorm.cpython-310.pyc
│   │   │   │   │       │   └── selective_state_update.cpython-310.pyc
│   │   │   │   │       └── selective_state_update.py
│   │   │   │   ├── [01;34m__pycache__[0m
│   │   │   │   │   └── __init__.cpython-310.pyc
│   │   │   │   └── [01;34mutils[0m
│   │   │   │       ├── generation.py
│   │   │   │       ├── hf.py
│   │   │   │       ├── __init__.py
│   │   │   │       └── [01;34m__pycache__[0m
│   │   │   │           ├── generation.cpython-310.pyc
│   │   │   │           ├── hf.cpython-310.pyc
│   │   │   │           └── __init__.cpython-310.pyc
│   │   │   ├── [01;34mmamba_ssm.egg-info[0m
│   │   │   │   ├── dependency_links.txt
│   │   │   │   ├── PKG-INFO
│   │   │   │   ├── requires.txt
│   │   │   │   ├── SOURCES.txt
│   │   │   │   └── top_level.txt
│   │   │   ├── README.md
│   │   │   ├── [01;32mselective_scan_cuda.cpython-310-x86_64-linux-gnu.so[0m
│   │   │   ├── setup.py
│   │   │   └── [01;34mtests[0m
│   │   │       └── [01;34mops[0m
│   │   │           ├── test_selective_scan.py
│   │   │           └── [01;34mtriton[0m
│   │   │               └── test_selective_state_update.py
│   │   ├── mambanet2.py
│   │   ├── mambanet.py
│   │   ├── models_mamba.py
│   │   ├── pure_mambanet.py
│   │   ├── [01;34m__pycache__[0m
│   │   │   ├── mambanet2.cpython-310.pyc
│   │   │   ├── mambanet.cpython-310.pyc
│   │   │   ├── models_mamba.cpython-310.pyc
│   │   │   ├── pure_mambanet.cpython-310.pyc
│   │   │   ├── rope.cpython-310.pyc
│   │   │   ├── udfe.cpython-310.pyc
│   │   │   └── vimnet.cpython-310.pyc
│   │   ├── rope.py
│   │   ├── test.py
│   │   ├── udfe.py
│   │   └── vimnet.py
│   ├── [01;34mmodule[0m
│   │   ├── __init__.py
│   │   ├── LightRFB.py
│   │   ├── [01;34mPNS[0m
│   │   │   ├── [01;34mPNS_Module[0m
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── reference.cpp
│   │   │   │   ├── reference.h
│   │   │   │   ├── sa.cu
│   │   │   │   ├── sa_ext.cpp
│   │   │   │   ├── timer.h
│   │   │   │   └── utils.h
│   │   │   └── setup.py
│   │   ├── PNSPlusModule.py
│   │   ├── PNSPlusNetwork.py
│   │   ├── [01;34m__pycache__[0m
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── LightRFB.cpython-310.pyc
│   │   │   ├── PNSPlusModule.cpython-310.pyc
│   │   │   ├── PNSPlusNetwork.cpython-310.pyc
│   │   │   └── Res2Net_v1b.cpython-310.pyc
│   │   └── Res2Net_v1b.py
│   ├── [01;34m__pycache__[0m
│   │   ├── __init__.cpython-310.pyc
│   │   └── __init__.cpython-36.pyc
│   ├── [01;34mutils[0m
│   │   ├── __init__.py
│   │   ├── [01;34m__pycache__[0m
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── __init__.cpython-36.pyc
│   │   │   ├── utils.cpython-310.pyc
│   │   │   └── utils.cpython-36.pyc
│   │   └── utils.py
│   └── [01;34mvmamba[0m
│       ├── csms6s.py
│       ├── csm_triton.py
│       ├── [01;34mmamba2[0m
│       │   ├── __init__.py
│       │   ├── k_activations.py
│       │   ├── layernorm_gated.py
│       │   ├── layer_norm.py
│       │   ├── [01;34m__pycache__[0m
│       │   │   ├── __init__.cpython-310.pyc
│       │   │   ├── k_activations.cpython-310.pyc
│       │   │   ├── layernorm_gated.cpython-310.pyc
│       │   │   ├── ssd_bmm.cpython-310.pyc
│       │   │   ├── ssd_chunk_scan.cpython-310.pyc
│       │   │   ├── ssd_chunk_state.cpython-310.pyc
│       │   │   ├── ssd_combined.cpython-310.pyc
│       │   │   ├── ssd_minimal.cpython-310.pyc
│       │   │   └── ssd_state_passing.cpython-310.pyc
│       │   ├── selective_state_update.py
│       │   ├── ssd_bmm.py
│       │   ├── ssd_chunk_scan.py
│       │   ├── ssd_chunk_state.py
│       │   ├── ssd_combined.py
│       │   ├── ssd_minimal.py
│       │   └── ssd_state_passing.py
│       ├── [01;34m__pycache__[0m
│       │   ├── csms6s.cpython-310.pyc
│       │   ├── csm_triton.cpython-310.pyc
│       │   ├── vmamba.cpython-310.pyc
│       │   └── vmambanet.cpython-310.pyc
│       ├── [01;34mselective_scan[0m
│       │   ├── [01;34mbuild[0m
│       │   │   ├── [01;34mbdist.linux-x86_64[0m
│       │   │   ├── [01;34mlib.linux-x86_64-cpython-310[0m
│       │   │   │   ├── selective_scan_cuda_core.cpython-310-x86_64-linux-gnu.so
│       │   │   │   ├── selective_scan_cuda_ndstate.cpython-310-x86_64-linux-gnu.so
│       │   │   │   └── selective_scan_cuda_oflex.cpython-310-x86_64-linux-gnu.so
│       │   │   └── [01;34mtemp.linux-x86_64-cpython-310[0m
│       │   │       ├── build.ninja
│       │   │       ├── [01;34mcsrc[0m
│       │   │       │   └── [01;34mselective_scan[0m
│       │   │       │       ├── [01;34mcus[0m
│       │   │       │       │   ├── selective_scan_core_bwd.o
│       │   │       │       │   ├── selective_scan_core_fwd.o
│       │   │       │       │   └── selective_scan.o
│       │   │       │       ├── [01;34mcusndstate[0m
│       │   │       │       │   ├── selective_scan_core_bwd.o
│       │   │       │       │   ├── selective_scan_core_fwd.o
│       │   │       │       │   └── selective_scan_ndstate.o
│       │   │       │       └── [01;34mcusoflex[0m
│       │   │       │           ├── selective_scan_core_bwd.o
│       │   │       │           ├── selective_scan_core_fwd.o
│       │   │       │           └── selective_scan_oflex.o
│       │   │       ├── .ninja_deps
│       │   │       └── .ninja_log
│       │   ├── [01;34mcsrc[0m
│       │   │   └── [01;34mselective_scan[0m
│       │   │       ├── cub_extra.cuh
│       │   │       ├── [01;34mcus[0m
│       │   │       │   ├── selective_scan_bwd_kernel.cuh
│       │   │       │   ├── selective_scan_core_bwd.cu
│       │   │       │   ├── selective_scan_core_fwd.cu
│       │   │       │   ├── selective_scan.cpp
│       │   │       │   └── selective_scan_fwd_kernel.cuh
│       │   │       ├── [01;34mcusndstate[0m
│       │   │       │   ├── selective_scan_bwd_kernel_ndstate.cuh
│       │   │       │   ├── selective_scan_core_bwd.cu
│       │   │       │   ├── selective_scan_core_fwd.cu
│       │   │       │   ├── selective_scan_fwd_kernel_ndstate.cuh
│       │   │       │   ├── selective_scan_ndstate.cpp
│       │   │       │   └── selective_scan_ndstate.h
│       │   │       ├── [01;34mcusnrow[0m
│       │   │       │   ├── selective_scan_bwd_kernel_nrow.cuh
│       │   │       │   ├── selective_scan_core_bwd2.cu
│       │   │       │   ├── selective_scan_core_bwd3.cu
│       │   │       │   ├── selective_scan_core_bwd4.cu
│       │   │       │   ├── selective_scan_core_bwd.cu
│       │   │       │   ├── selective_scan_core_fwd2.cu
│       │   │       │   ├── selective_scan_core_fwd3.cu
│       │   │       │   ├── selective_scan_core_fwd4.cu
│       │   │       │   ├── selective_scan_core_fwd.cu
│       │   │       │   ├── selective_scan_fwd_kernel_nrow.cuh
│       │   │       │   └── selective_scan_nrow.cpp
│       │   │       ├── [01;34mcusoflex[0m
│       │   │       │   ├── selective_scan_bwd_kernel_oflex.cuh
│       │   │       │   ├── selective_scan_core_bwd.cu
│       │   │       │   ├── selective_scan_core_fwd.cu
│       │   │       │   ├── selective_scan_fwd_kernel_oflex.cuh
│       │   │       │   └── selective_scan_oflex.cpp
│       │   │       ├── reverse_scan.cuh
│       │   │       ├── selective_scan_common.h
│       │   │       ├── selective_scan.h
│       │   │       ├── static_switch.h
│       │   │       └── uninitialized_copy.cuh
│       │   ├── README.md
│       │   ├── [01;34mselective_scan.egg-info[0m
│       │   │   ├── dependency_links.txt
│       │   │   ├── PKG-INFO
│       │   │   ├── requires.txt
│       │   │   ├── SOURCES.txt
│       │   │   └── top_level.txt
│       │   ├── setup.py
│       │   ├── test_selective_scan_easy.py
│       │   ├── test_selective_scan.py
│       │   └── test_selective_scan_speed.py
│       ├── vmambanet.py
│       └── vmamba.py
├── README.md
└── [01;34mscripts[0m
    ├── add_image.py
    ├── config.py
    ├── eval_eff.py
    ├── my_test.py
    ├── my_train.py
    ├── post_test.py
    └── [01;34m__pycache__[0m
        ├── config.cpython-310.pyc
        └── config.cpython-36.pyc

87 directories, 484 files
