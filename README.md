# 15740Project

We focus on two operations: convolution and matrix-vector multiplication.

## The General Architecture
All the components below are independent to each other and can work in parallel:
- **DSP:** DSP is the function unit that performs floating point computation. The chip has `D` DSPs which can work in parallel. Each floating point computation takes `C` cycles.
- **Memory:** Memory is separated into on-chip and off-chip memory. Data must be read onto the chip before it can be used by DSPs. Reading `k` bytes from off-chip to on-chip needs `A + B * k` cycles, where `A` and `B` are constants. The size of the on-chip memory is `M` bytes. The off-chip memory is infinitely large.
- **Arrays:** To use the on-chip memory, construct on-chip arrays. DSPs can directly access on-chip arrays taking no cycle (it's already included in the `C` cycles), but *array indexing also takes `X` cycles*.
- **Flip-flops:** Flip-flops can be used as state registers between two modules. Writing to a flip-flop does not change the value it stores until `chip.join()` is called.
- **Ports:** There is only one read-write port between on-chip and off-chip. Each array only has one read-write port, so only one module can access it at a time.


## Chip Operations
- `chip = Chip(...)`: Constructs a chip
- `ff = chip.flip_flop()`: Constructs a flip-flop that can store a floating point number. Read the data with `ff.read()` and write data to it with `ff.write(data)`. The new value will not be written until `chip.join()` is called. If no new value is written, then the old value will be kept. Reading/writing a flip-flop does not take any cycle. Flip-flops are mainly used to pipeline two modules. (It is unnecessary to implement registers only used by one module as flip-flops.)
- `onchip_array = chip.array(num, name)`: Allocates an on-chip float array of `num` items. Use `get_item()` to access the items in the array.
- `chip.read(onchip_array, onchip_offset, offchip_array, offchip_offset, num)`: Copies a chunk of data from off-chip to on-chip. Takes `A + B * (4 * num)` cycles.
- `chip.read_and_partition(array_offset_list, offchip_array, offchip_offset, num, mode)`: Copies a chunk of data from off-chip to several on-chip arrays. Takes `A + B * (4 * num)` cycles. `array_offset_list` is a list of `(onchip_array, onchip_offset)`. `mode` can be either `'block'` or `'cycle'`. Zero-padding will be done automatically.
- `chip.write(onchip_array, onchip_offset, offchip_array, offchip_offset, num)`: Copies a chunk of float data from on-chip to off-chip. Takes `A + B * (4 * num)` cycles.
- `chip.get_item(onchip_array, index)`: Returns `onchip_array[index]`. Takes `X` cycles for indexing.
- `chip.compute(value)`: Returns the same value, but uses one DSP to compute and takes `C` cycles.
- `chip.array_write(dst_array, dst_offset, value)`: Writes the value back to an on-chip array. Takes `X` cycles for indexing.

## Modules
A module consists of an array of operations which run *sequentially*. Multiple modules run *in parallel* until they meet the `chip.join()` function.
- `module = chip.module(func, name)`: Construct a module that runs the function `func`. Call this module with `ans = module(...)`.
- `chip.join()`: Join all active modules and update the number of cycles. This method separates the code into blocks. Modules in the same block run in parallel. Modules in different blocks never run at the same time. In the same block, only one module can access the port, and each array can only be accessed by one array.

The best approach is to **put all operations into modules**. If an operation does not belong to any module, an auto module will be created for this single operation.

## Loop Unrolling
- `chip.unrolled_loop(n, f)`: Fully unroll a loop that iterates `n` times. The function `f` takes an argument `idx`, which is its index in the loop.

## Example

`EIE.py` contains the implementation of EIE. Run this example with the following command:
```
python EIE.py -A 1 -B 0.01 -C 1 -D 100 -M 10000000 -X 1 -P 4 -n 8 -s 0.5
```


`spmv.py` contains an example of SPMV. Run this example with the following command:
```
python spmv.py -A 1 -B 0.1 -C 1 -X 1 -D 1 -M 100000
```
