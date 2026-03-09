<!-- Ascend C 算子 API 参考：供 kernel 生成时拼入 prompt。内容整理自昇腾 CANN 8.0.0 开发文档，仅供参考与评测。 -->

## 说明

本参考用于 Ascend C 核函数（Kernel）生成，内容来自昇腾社区 CANN 8.0.0 开发文档与 Ascend C API 参考，仅保留与设备侧核函数编程相关的接口与约定。

---

## 核函数与编程模型

核函数是 Ascend C 算子设备侧实现的入口。多个核会执行相同的核函数代码，通过 block_idx 区分。

**定义规则：**

- 使用函数类型限定符 `__global__` 和 `__aicore__`：`__global__` 表示可被 `<<<>>>` 调用，`__aicore__` 表示在设备端 AI Core 上执行。
- 指针入参需加变量类型限定符 `__gm__`，表示指向 Global Memory。可选使用宏：`#define GM_ADDR __gm__ uint8_t*`，入参写为 `GM_ADDR x` 等，后续再转为实际类型。
- 仅支持入参为指针或 C/C++ 内置类型（如 `half*`, `float*`, `int32_t`）；返回类型必须为 `void`。

**定义示例：**

```cpp
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z)
{
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}
```

**调用方式：**

```cpp
kernel_name<<<blockDim, l2ctrl, stream>>>(argument list);
```

- `blockDim`：在多少个核上执行，每个核有逻辑 ID（block_idx），核内通过 `AscendC::GetBlockIdx()` 获取。
- 调用为异步，主机端可用 `aclrtSynchronizeStream(stream)` 等待执行完毕。

**典型结构：** Init（设置 GlobalTensor、初始化 Buffer/Queue） + Process（CopyIn → Compute → CopyOut）。

---

## 基本数据结构与内存

**GlobalTensor**：表示全局内存上的数据。使用前用 `SetGlobalBuffer` 绑定 GM 指针与元素个数：

```cpp
AscendC::GlobalTensor<float> xGm;
xGm.SetGlobalBuffer((__gm__ float*)x + offset, blockLength);
```

**LocalTensor**：表示 AI Core 本地内存上的数据，用于 VECIN、VECOUT、VECCALC 等位置。一般从队列的 `AllocTensor<T>()` 获得。

**TPipe**：管理全局内存等资源的框架，用于为队列分配 Buffer：

```cpp
AscendC::TPipe pipe;
pipe.InitBuffer(inQueueX, 1, 512 * sizeof(float));  // bufferNum=1, sizeInBytes
```

**TQue（队列）**：用于 CopyIn、Compute、CopyOut 之间的任务同步与内存管理。常用位置：`QuePosition::VECIN`、`QuePosition::VECOUT`（部分代码或文档中写作 `TPosition::VECIN`/`VECOUT`，等价）。

- `AllocTensor<T>()`：从队列分配一块 LocalTensor。
- `FreeTensor(localTensor)`：释放该块回队列。
- `EnQue(localTensor)` / `EnQue<T>(localTensor)`：入队。
- `DeQue<T>()`：出队，得到 LocalTensor。

示例类型声明：

```cpp
AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
```

---

## 数据搬运 API

### DataCopy（GM 与 Local 之间）

- **GM → Local（连续）**：`AscendC::DataCopy(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t calCount);`
- **Local → GM（连续）**：`AscendC::DataCopy(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal, const uint32_t calCount);`

`calCount` 为参与搬运的**元素个数**。要求 `calCount * sizeof(T)` 为 32 字节的倍数，否则搬运量按 32 字节向下取整。

不连续搬运使用 `DataCopyParams`：

```cpp
struct DataCopyParams {
    uint16_t blockCount = 0;  // 连续传输数据块个数 [1, 4095]
    uint16_t blockLen = 0;    // 每块长度，单位 datablock(32B)
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
};
```

对应原型：`DataCopy(dstLocal, srcGlobal, repeatParams)` 或 `DataCopy(dstGlobal, srcLocal, repeatParams)`。

### Copy（Local 与 Local：VECIN/VECCALC/VECOUT 之间）

VECIN、VECCALC、VECOUT 之间的搬运，支持 mask 与 datablock 步长：

```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void Copy(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const uint8_t repeatTimes, const CopyRepeatParams& repeatParams);
// 或 mask 连续模式：const uint64_t mask
```

`CopyRepeatParams`：`dstStride`, `srcStride`, `dstRepeatSize`, `srcRepeatSize`（同一迭代内 datablock 步长、相邻迭代间步长）。LocalTensor 起始地址需 32 字节对齐。

---

## 常用计算 API

计算基于 LocalTensor，在 Vector 计算单元上执行。以下为常用向量/标量 API（Ascend C 基础 API，CANN 8.0.0）。

- **双目（前 n 个元素）**：`Add(dst, src1, src2, n)`、`Sub`、`Mul`、`Div`、`Max`、`Min`、`And`、`Or`；`AddRelu`、`SubRelu`；`MulCast`、`FusedMulAdd`、`FusedMulAddRelu` 等。
- **单目**：`Exp`、`Ln`、`Abs`、`Reciprocal`、`Sqrt`、`Rsqrt`、`Not`、`Relu`。
- **标量+向量**：`Adds(dst, src, scalar, n)`、`Muls`、`Maxs`、`Mins`；`ShiftLeft`、`ShiftRight`；`LeakyRelu`。
- **三目**：`Axpy`（标量乘加）。
- **比较**：`Compare`、`CompareScalar`；**选择**：`Select`；**归约**：`ReduceMax`、`ReduceMin`、`ReduceSum`；**精度转换**：`Cast`。

也可用运算符重载做整个 tensor 参与计算：`dst = src1 + src2`（支持 `+ - * / | & < > <= >= == !=`）。

**多核**：核内通过 `AscendC::GetBlockIdx()` 获取当前 block 逻辑 ID，用于按块切分 GM 偏移（如 `offset = blockLength * GetBlockIdx()`）。`GetBlockNum()` 可获取 block 总数（若 API 提供）。

---

## 算子接口约定

- Host 侧使用 `kernel_name<<<blockDim, l2ctrl, stream>>>(...)` 调用核函数；核内用 `GetBlockIdx()` 做多核分块（如按 block 划分 GM 区间）。
- 典型 Kernel 类结构：
  - **Init**：对 GM 指针用 `SetGlobalBuffer` 绑定各 GlobalTensor；`pipe.InitBuffer` 为各 TQue 分配 buffer。
  - **Process**：依次调用 CopyIn、Compute、CopyOut。
  - **CopyIn**：`AllocTensor` → `DataCopy(GM→Local)` → `EnQue`。
  - **Compute**：`DeQue` 取输入 LocalTensor，`AllocTensor` 取输出 buffer，调用计算 API（如 Add），再 `EnQue` 输出、`FreeTensor` 输入。
  - **CopyOut**：`DeQue` 取结果 LocalTensor，`DataCopy(Local→GM)`，`FreeTensor`。

头文件：一般包含 `kernel_operator.h`；API 位于 Ascend C 类库，命名空间为 `AscendC`。

---

## 引用

以上内容整理自昇腾社区 **CANN 8.0.0 开发文档**与 **Ascend C API 参考**（hiascend.com），仅供 kernel 生成与评测参考。
