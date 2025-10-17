---
pipeline_tag: MultiModal
frameworks:
  - PyTorch
license: apache-2.0
hardwares:
  - NPU
language:
  - en
---

## 注意
- 本模型仓代码，是针对此开源链接进行的适配
https://github.com/Tencent-Hunyuan/HunyuanImage-3.0

## 一、准备运行环境

  **表 1**  版本配套表

| 配套  | 版本 | 环境准备指导 |
| ----- | ----- |-----|
| Python | 3.10. | - |
| torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I A2(8*64G)推理设备：支持的卡数最小为1
- [Atlas 800I A2(8*64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 1.3 环境依赖安装
```shell
pip3 install -r requirements.txt
```

### 1.4 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.5 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

## 二、下载权重

### 2.1 权重及配置文件说明
HunyuanImage-3.0 模型权重链接：

```shell
	https://huggingface.co/tencent/HunyuanImage-3.0
```
## 三、HunyuanImage-3.0 使用

当前支持的卡数：1、2、4、8、16
### 3.1 下载到本地
```shell
   git clone https://modelers.cn/MindIE/HunyuanImage-3.0.git
   cd HunyuanImage-3.0
```
### 3.2 单卡性能测试

#### 3.2.1 等价优化

执行命令：
```shell
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expand_segements:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

export ASCEND_LAUNCH_BLOCKING=1

torchrun --nproc_per_node=${WORLD_SIZE} run_image_gen_tp.py \
         --model-id /data/weights/HunyuanImage-3.0 \
         --verbose 1 \
         --sys-deepseek-prompt "universal" \
         --prompt "A brown and white dog is running on the grass" \
         --image-size 512x512 \
         --diff-infer-steps 50 \
         --seed 1234 \
         --reproduce
```
参数说明：
| 参数                    | 说明                                                         | 默认值      |
| ----------------------- | ------------------------------------------------------------ | ----------- |
| `ALGO`                  | 为0表示默认FA算子；设置为1表示使用高性能FA算子               |             |
| `nproc_per_node`        | 并行推理的总卡数                                             |             |
| `--prompt`              | 输入的提示词                                                 | (必填)      |
| `--model-id`            | 模型权重路径                                                 | (必填)      |
| `--attn-impl`           | attention 实现方式，可选 `sdpa` 或 `flash_attention_2`       | `sdpa`      |
| `--moe-impl`            | MoE 实现方式，可选 `eager` 或 `flashinfer`                   | `eager`     |
| `--seed`                | 生图的随机种子                                               | `None`      |
| `--diff-infer-steps`    | 采样步数                                                     | `50`        |
| `--image-size`          | 生成图像的分辨率，可选 `auto`, `1280x768` 或 `16:9`          | `512x512`   |
| `--save`                | 保存生成图像的路径                                           | `image.png` |
| `--verbose`             | 日志打印等级，0: 不打印，1: 打印推理信息                     | `0`         |
| `--rewrite`             | 选择是否开启提示词改写，默认关闭                             | `0`         |
| `--sys-deepseek-prompt` | 选择 `universal` 或者 `text_rendering`指向的提示词作为系统提示词 | `universal` |

### 3.3 8卡性能测试

#### 3.3.1 8卡 TP 性能测试

执行命令：
```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export HCCL_DETERMINISTIC=true

export ASCEND_LAUNCH_BLOCKING=1

torchrun --nproc_per_node=${WORLD_SIZE} run_image_gen_tp.py \
         --model-id /data/weights/HunyuanImage-3.0 \
         --verbose 1 \
         --sys-deepseek-prompt "universal" \
         --prompt "A brown and white dog is running on the grass" \
         --image-size 512x512 \
         --diff-infer-steps 50 \
         --seed 1234 \
         --reproduce
```
参数说明： 
| 参数                    | 说明                                                         | 默认值      |
| ----------------------- | ------------------------------------------------------------ | ----------- |
| `ALGO`                  | 为0表示默认FA算子；设置为1表示使用高性能FA算子               |             |
| `nproc_per_node`        | 并行推理的总卡数                                             |             |
| `--prompt`              | 输入的提示词                                                 | (必填)      |
| `--model-id`            | 模型权重路径                                                 | (必填)      |
| `--attn-impl`           | attention 实现方式，可选 `sdpa` 或 `flash_attention_2`       | `sdpa`      |
| `--moe-impl`            | MoE 实现方式，可选 `eager` 或 `flashinfer`                   | `eager`     |
| `--seed`                | 生图的随机种子                                               | `None`      |
| `--diff-infer-steps`    | 采样步数                                                     | `50`        |
| `--image-size`          | 生成图像的分辨率，可选 `auto`, `1280x768` 或 `16:9`          | `512x512`   |
| `--save`                | 保存生成图像的路径                                           | `image.png` |
| `--verbose`             | 日志打印等级，0: 不打印，1: 打印推理信息                     | `0`         |
| `--rewrite`             | 选择是否开启提示词改写，默认关闭                             | `0`         |
| `--sys-deepseek-prompt` | 选择 `universal` 或者 `text_rendering`指向的提示词作为系统提示词 | `universal` |


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。
