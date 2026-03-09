# Ascend C `selected_shot` 样例代码硬编码与数据对齐问题分析

在 `prompts/ascendc_new_model_leaky_relu.py` 提供的 `selected_shot` 样例代码中，存在严重的硬编码问题。这些硬编码不仅极大地限制了 LLM 泛化生成其他算子的能力，更严重的是，它在 Ascend C 最核心的 **32 字节数据对齐（32-byte alignment）处理上存在逻辑漏洞，给 LLM 提供了一个错误的示范**。

以下是具体的硬编码点、导致的问题以及对 LLM 泛化能力和数据对齐理解的影响分析：

## 1. 硬编码点分析及对 LLM 泛化能力的限制

在样例的 `host_operator_src` 中，切分（Tiling）逻辑写死了几个常数：
```cpp
const uint32_t BLOCK_DIM = 48;
const uint32_t ALIGN_NUM = 8;
const uint32_t MAX_TILE_LENGTH = 8192; 
```

### ① `BLOCK_DIM = 48`（硬编码核数）
* **带来的问题**：Ascend910B 虽多为 48 核心，但写死之后没有利用动态查询的方法获取最大可用核数。当输入数据量较小（比如 `totalLength < 48 * 8 = 384` 时），通过 `blockLength = (blockLength / ALIGN_NUM) * ALIGN_NUM` 的下取整，**前 47 个核分配到的数据量会全部变成 0**，最后所有的计算任务全部堆积在最后一个核心上（核心 47 处理所有的 `totalLength`）。
* **泛化影响**：LLM 学习到了这种“无脑固定切 48 块”的简单一维平铺逻辑。遇到 Softmax、Reduce 等需要跨维度处理或需要灵活分配任务的算子时，LLM 会误以为所有算子都可以直接 `totalLength / 48`，完全缺乏处理复杂张量维度的能力。

### ② `ALIGN_NUM = 8`（硬编码数据对齐个数）
* **带来的问题**：样例是 `float` 类型（4字节）。Ascend C 的 `DataCopy` 和大部分矢量指令要求内存和搬运长度必须是 **32字节对齐**。因此 `32 字节 / 4 字节(float) = 8 个元素`，这里的 `8` 是算出来的。
* **泛化影响**：因为样例代码里没有体现出 `ALIGN_NUM` 和 `sizeof(DType)` 的关系，LLM 在泛化时根本不理解 `8` 是怎么来的。如果让 LLM 编写 `half` (2字节) 或 `int8` (1字节) 的算子，它极大概率会原封不动地照抄 `ALIGN_NUM = 8`，导致实际只对齐了 16 字节或 8 字节，从而在真机上运行时直接触发未对齐异常（总线错误）。

### ③ `MAX_TILE_LENGTH = 8192`（硬编码 UB 单次处理上限）
* **带来的问题**：8192 个 float 占用约 32KB。样例比较简单，分配了几个 `TBuf` 和 `TQue`，没有超过 Unified Buffer (UB) 的容量限制（例如 256KB）。
* **泛化影响**：如果 LLM 生成的算子稍微复杂一点（比如需要额外申请四五个中间 Tensor 用于保存复杂的中间态，或者在 `double` 数据类型下），它依旧死板套用 `tileLength = 8192`，会直接导致 UB 显存溢出，算子无法编译或运行。

---

## 2. Ascend C 32 字节数据对齐的致命误导（最核心问题）

样例本身在处理 **尾部数据对齐** 时存在严重的 Bug。我们来看样例的 Tiling 切分逻辑：

```cpp
    uint32_t blockLength = totalLength / BLOCK_DIM;
    blockLength = (blockLength / ALIGN_NUM) * ALIGN_NUM; // 这里强行对齐了 8 的倍数
    uint32_t lastBlockLength = totalLength - blockLength * (BLOCK_DIM - 1);
```

这段逻辑保护了前 `N-1` 个核心（`blockLength` 确实是 8 的倍数，满足 32 字节要求），**但是完全没有保护 `lastBlockLength`！**

如果用户传入的一个 Tensor `totalLength = 100`。
* 100 不是 8 的倍数。算出的 `lastBlockLength` 也绝对不是 8 的倍数。
* 在 `KernelLeakyRelu::Process` 处理到尾部时，最后一段 `tailLength` 同样不会是 8 的倍数。
* 当核函数调用 `AscendC::DataCopy(xLocal, xGm[...], length)` 时，由于传入的 `length` 对应的字节数不是 32 字节对齐的整数倍，底层硬件将无法直接执行对齐拷贝！

### 对 LLM 造成的毁灭性误导：
1. **传授了错误的编程直觉**：LLM 看到这个样例，会认为 "哦，Ascend C 里只要对 `blockLength` 对齐就行了，尾部的 `lastBlockLength` 哪怕不对齐，直接丢进 `DataCopy` 和 `AscendC::Maxs/Mins` 里也是可以正常运行的"。
2. **泛化算子必然崩溃**：一旦 LLM 吸收了这个“知识”，生成的任何新算子只要遇到任意非 32 字节倍数形状的 Tensor，都会因为尾部（tail）非对齐而在 Ascend NPU 上触发 **Bus Error（总线错误）或非法的越界内存访问**。

### 正确的解决方案（LLM 应该学到却没有学到的）
在 Ascend C 中，处理不规则长度的尾块（Tail block）需要告诉 LLM 使用正确的对齐策略，例如：
1. **向上对齐拷贝**：将尾部数据长度向上补齐到 32 字节（8个float），从 Global Memory 拷贝到 UB 后，在计算时通过多余不计算或 Mask 的方式屏蔽无效数据，然后再将实际有效长度拷贝回（写回时也可能涉及复杂操作）。
2. **标量操作托底（针对尾部极其不规则部分）**：将多余的未对齐尾部元素隔离出来，在 Kernel 侧使用标量运算 (Scalar) 逐个循环搬运和计算，避免直接调用矢量的 `DataCopy`。

## 总结
当前的 Selected Shot 给了 LLM 很多“图省事”的硬编码参数，没有展示出处理数据类型的动态宏观视野。最致命的是，它本身就是一个**残缺的数据对齐示例**。它会让 LLM 丧失跨数据类型的泛化能力，并在处理不规则 shape 张量时，写出必然导致硬件崩溃的劣质代码。