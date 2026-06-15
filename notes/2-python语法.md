# Python 语法笔记

本文记录阅读 DeepEP 源码时遇到的 Python 语法点。每个语法点按“作用、基础例子、DeepEP 中的用法、总结”组织。

## 1. yield

### 1.1 作用

`yield` 用在函数内部，用来定义一个生成器函数（generator function）。

普通函数执行到 `return` 会一次性返回结果并结束；生成器函数执行到 `yield` 时，会把一个值“产出”给调用方，并暂停在当前位置。下一次继续迭代时，会从暂停的位置继续执行。

### 1.2 基础例子

```python
def numbers():
    yield 1
    yield 2
    yield 3

for x in numbers():
    print(x)
```

输出：

```text
1
2
3
```

这里 `numbers()` 不会一次性返回 `[1, 2, 3]`，而是返回一个 generator。每次循环取值时，函数才继续向下执行到下一个 `yield`。

### 1.3 yield 和 return 的区别

```python
def use_return():
    return [1, 2, 3]

def use_yield():
    yield 1
    yield 2
    yield 3
```

| 对比项 | `return` | `yield` |
| --- | --- | --- |
| 返回方式 | 一次性返回完整结果 | 每次产出一个结果 |
| 函数状态 | 返回后函数结束 | 产出后函数暂停 |
| 调用结果 | 普通对象 | generator 对象 |
| 适合场景 | 结果较小、一次性计算 | 按需生成、组合枚举、节省内存 |

### 1.4 为什么用 yield

`yield` 适合以下场景：

- 结果很多，不想一次性全部放进内存。
- 需要按需生成数据。
- 需要表达一个可迭代的测试组合、配置组合或数据流。

例如：

```python
def modes():
    for fp8 in (0, 1):
        for async_mode in (0, 1):
            yield fp8, async_mode

for fp8, async_mode in modes():
    print(fp8, async_mode)
```

这会依次生成 4 组组合：

```text
0 0
0 1
1 0
1 1
```

### 1.5 DeepEP 中的用法

在 `tests/elastic/test_ep.py` 中，`enumerate_ep_modes()` 使用了 `yield`：

```python
def enumerate_ep_modes():
    for do_handle_copy in (1, 0):
        for expert_alignment in (128, 1):
            for use_fp8_dispatch in (1, 0):
                for num_bias in (0, 1, 2):
                    for with_previous_event in (0, 1):
                        for async_with_compute_stream in (0, 1):
                            for allocate_on_comm_stream in ((1, ) if with_previous_event else (0, 1)):
                                yield (do_handle_copy, expert_alignment, use_fp8_dispatch, num_bias,
                                       with_previous_event, async_with_compute_stream, allocate_on_comm_stream)
```

这个函数的作用是枚举 EP 测试的所有模式组合。每次循环到最内层时，`yield` 产出一组配置：

```text
do_handle_copy
expert_alignment
use_fp8_dispatch
num_bias
with_previous_event
async_with_compute_stream
allocate_on_comm_stream
```

调用方这样使用：

```python
for (do_handle_copy, expert_alignment, use_fp8_dispatch, num_bias,
     with_previous_event, async_with_compute_stream, allocate_on_comm_stream) in enumerate_ep_modes():
    ...
```

这里不需要先构造一个巨大的 list。测试代码可以一边生成配置，一边执行测试。

### 1.6 嵌套 for 循环如何和 yield 配合

在 DeepEP 的例子中，`yield` 写在最内层循环里，但所有外层 `for` 循环都会被保留在 generator 的执行状态中。

执行逻辑是：

1. 进入最外层 `for do_handle_copy in (1, 0)`。
2. 继续进入下一层 `for expert_alignment in (128, 1)`。
3. 一直进入到最内层循环。
4. 执行 `yield (...)`，产出当前所有循环变量组成的一组配置。
5. 函数暂停，所有循环位置都被保存。
6. 下一次迭代时，从 `yield` 后面继续。
7. 如果最内层循环还有下一个值，就继续产出。
8. 如果最内层循环结束，就回到上一层循环推进。
9. 直到所有外层循环都结束。

简化例子：

```python
def modes():
    for a in (1, 2):
        for b in ("x", "y"):
            yield a, b
```

它会产出：

```text
1, x
1, y
2, x
2, y
```

所以，虽然 `yield` 只写在最内层，外层循环仍然完整参与组合枚举。

### 1.7 手动迭代 generator

生成器也可以用 `next()` 手动取值：

```python
g = numbers()

print(next(g))  # 1
print(next(g))  # 2
print(next(g))  # 3
```

当生成器没有更多值时，再调用 `next(g)` 会抛出 `StopIteration`。

### 1.8 小结

`yield` 用来把函数变成生成器，让函数可以“暂停并产出一个值”，下次迭代时再从暂停位置继续执行。它常用于按需生成数据、节省内存，以及枚举测试配置组合。

## 2. if __name__ == '__main__':

### 2.1 作用

`if __name__ == '__main__':` 是 Python 脚本里常见的入口判断。

它的作用是区分两种情况：

- 当前文件是被直接运行的。
- 当前文件是被其他 Python 文件 import 的。

Python 每个模块都有一个内置变量 `__name__`：

- 如果文件被直接运行，`__name__` 的值是 `'__main__'`。
- 如果文件被 import，`__name__` 的值是模块名。

### 2.2 基础例子

假设有一个文件 `demo.py`：

```python
def main():
    print("run main")

if __name__ == '__main__':
    main()
```

直接运行：

```bash
python demo.py
```

此时 `__name__ == '__main__'` 成立，会执行 `main()`。

如果被其他文件 import：

```python
import demo
```

此时 `demo.py` 中的函数定义会被加载，但 `__name__ == '__main__'` 不成立，所以不会自动执行 `main()`。

### 2.3 为什么需要它

这个写法可以让同一个文件同时具备两种用途：

- 作为脚本直接运行时，执行入口逻辑。
- 作为模块被 import 时，只提供函数、类、常量，不自动执行测试、启动进程或解析命令行参数。

这在测试脚本、工具脚本中很常见。

### 2.4 DeepEP 中的用法

在 `tests/elastic/test_ep.py` 文件末尾有：

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test elastic EP kernels')
    ...
    args = parser.parse_args()
    ...
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
```

这表示：

- 当直接运行 `python tests/elastic/test_ep.py` 时，会解析命令行参数，并通过 `torch.multiprocessing.spawn` 启动多进程测试。
- 当其他文件 import `test_ep.py` 时，只会加载其中的函数，例如 `enumerate_ep_modes()`、`test_dispatch_combine()`、`test_loop()`，不会立刻启动多进程测试。

这很重要，因为 `test_ep.py` 会创建分布式进程、初始化 GPU 通信和运行大量测试。如果没有 `if __name__ == '__main__':` 保护，其他文件只要 import 它，就可能意外启动测试流程。

### 2.5 小结

`if __name__ == '__main__':` 用来声明“只有当前文件被直接运行时，才执行这里的入口逻辑；被 import 时不要执行”。

## 3. 模块

### 3.1 作用

Python 中的模块（module）可以简单理解为一个 `.py` 文件。

例如有一个文件：

```text
math_utils.py
```

那么它就是一个 Python 模块，模块名通常就是文件名去掉 `.py` 后缀：

```text
math_utils
```

模块的作用是组织代码。一个模块里可以放：

- 函数
- 类
- 常量
- 可执行语句
- import 语句

### 3.2 基础例子

假设有一个文件 `math_utils.py`：

```python
def add(a, b):
    return a + b

PI = 3.14159
```

另一个文件可以这样使用它：

```python
import math_utils

print(math_utils.add(1, 2))
print(math_utils.PI)
```

也可以只导入其中一部分：

```python
from math_utils import add

print(add(1, 2))
```

### 3.3 模块和包的关系

如果一个目录里包含 Python 文件，并且通常带有 `__init__.py`，这个目录就可以作为包（package）使用。

简单理解：

- 单个 `.py` 文件是模块。
- 多个模块组成一个目录，这个目录可以是包。
- 包可以继续包含子包和子模块。

例如：

```text
deep_ep/
├── __init__.py
├── buffers/
│   ├── __init__.py
│   ├── elastic.py
│   └── legacy.py
└── utils/
    ├── __init__.py
    ├── envs.py
    └── math.py
```

这里：

- `deep_ep` 是一个包。
- `deep_ep.buffers.elastic` 是一个模块。
- `deep_ep.buffers.legacy` 是一个模块。
- `deep_ep.utils.envs` 是一个模块。
- `deep_ep.utils.math` 是一个模块。

### 3.4 DeepEP 中的用法

在 `tests/elastic/test_ep.py` 中有：

```python
import deep_ep
from deep_ep.utils.math import (
    align, count_bytes, calc_diff,
    per_token_cast_back, per_token_cast_to_fp8,
    safe_div
)
from deep_ep.utils.gate import get_unbalanced_scores
from deep_ep.utils.envs import init_dist, init_seed, dist_print
```

这些 import 的含义是：

- `import deep_ep`：导入整个 `deep_ep` 包，可以使用 `deep_ep.ElasticBuffer` 等入口 API。
- `from deep_ep.utils.math import align`：从 `deep_ep.utils.math` 模块中导入 `align` 函数。
- `from deep_ep.utils.gate import get_unbalanced_scores`：从 gate 工具模块中导入构造 MoE 路由分布的函数。
- `from deep_ep.utils.envs import init_dist`：从环境工具模块中导入分布式初始化函数。

### 3.5 模块导入时会发生什么

当 Python import 一个模块时，会执行该模块的顶层代码。

例如：

```python
# demo.py
print("loading demo")

def hello():
    print("hello")
```

当执行：

```python
import demo
```

会先输出：

```text
loading demo
```

因为 `print("loading demo")` 是模块顶层代码。

这也是为什么测试脚本常配合使用：

```python
if __name__ == '__main__':
    ...
```

这样可以避免模块被 import 时自动执行入口逻辑。

### 3.6 小结

Python 模块就是组织代码的基本单位，通常对应一个 `.py` 文件。通过 `import` 可以复用其他模块里的函数、类和常量。多个模块可以组织成包，例如 DeepEP 的 `deep_ep` 包。模块被 import 时会执行顶层代码，所以脚本入口通常需要用 `if __name__ == '__main__':` 保护。
