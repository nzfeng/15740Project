import numpy as np
import math
import sys


class ChipArray(object):
  UNUSED = 0
  READ = 1
  WRITE = 2
  COMPUTE = 3

  def __init__(self, num: int, name: str):
    self.num = num
    self.name = name
    self.data = np.zeros((num,), dtype=np.float)
    self.mode = ChipArray.UNUSED
    self.module_owner = None

  def __getitem__(self, index):
    raise RuntimeError('Cannot access an array item directly. Must use Chip.get_item()')

  def clear(self):
    self.data.fill(0)


class ChipFlipFlop(object):
  def __init__(self):
    self.next_data = 0.
    self.data = 0.
    self.written = False

  def read(self):
    return self.data

  def write(self, data: float):
    if self.written:
      sys.stderr.write('A flip-flop is written twice before calling join().\n')
    self.next_data = data
    self.written = True

  def flip(self):
    self.data = self.next_data
    self.written = False


class Chip(object):
  def __init__(self, dsp_num: int, dsp_cycle: int, index_cycle: int,
               mem_size: int, mem_overhead: int, mem_unit: float):
    self.dsp_num = dsp_num
    self.dsp_cycle = dsp_cycle
    self.index_cycle = index_cycle
    self.mem_num = mem_size // 4  # Float is 4 bytes
    self.mem_overhead = mem_overhead
    self.mem_unit = mem_unit

    self.array_list = []
    self.free_mem_num = self.mem_num
    self.port_owner = None  # Port between on-chip and off-chip

    self.current_module = None
    self.module_list = []
    self.array_accessed = [] # Arrays accessed by the current method

    self.ff_list = []
    self.dsp_list = []

    # #1: Module group stack
    self.module_group_stack = []

    # For analysis
    self.verbose = False
    self.module_history = []
    self.join_index = []
    self.cycles = 0

  def flip_flop(self) -> ChipFlipFlop:
    ff = ChipFlipFlop()
    self.ff_list.append(ff)
    return ff

  def array(self, num: int, name: str='') -> ChipArray:
    '''
    Allocate an on-chip array
    '''
    if num > self.free_mem_num:
      raise MemoryError('Out of on-chip memory.')

    self.free_mem_num -= num
    a = ChipArray(num, name)
    self.array_list.append(a)
    return a

  def _is_onchip(self, onchip_array: ChipArray) -> bool:
    idx = -1
    l = len(self.array_list)
    for i in range(l):
      if self.array_list[i] == onchip_array:
        idx = i
        break
    return idx > -1

  def _check_onchip(self, array_list: list):
    for a in array_list:
      if not self._is_onchip(a):
        raise MemoryError('Array {} is not on this chip.'.format(a.name))

  def _add_cycles(self, cycles: int):
    self.current_module.cycles += cycles

  def set_array_mode(self, a: ChipArray, mode: int):
    if not self._is_onchip(a):
      raise MemoryError('Array {} is not on this chip.'.format(a.name))
    if a.module_owner is None:
      if a.mode not in [mode, ChipArray.UNUSED]:
        raise RuntimeError('Array {} used for multiple purposes.'.format(a.name))
      a.mode = mode
      a.module_owner = self.current_module
      self.array_accessed.append(a)
    elif a.module_owner != self.current_module:
      raise RuntimeError('Array {} belongs to multiple modules.'.format(a.name))

  def _require_module(use_port: bool):
    def wrapper(func):
      def inner(self, *args, **kwargs):
        self.array_accessed = []
        p = False
        if self.current_module is None:
          # Construct an auto module for func.
          l = len(self.module_list)
          module = self.module(use_port, [], 'auto_module_{}'.format(l))
          self.current_module = module
          ans = func(self, *args, **kwargs)
          p = True

        ans = func(self, *args, **kwargs)
        # Check the port
        if use_port:
          if self.port_owner is None:
            self.port_owner = self.current_module
            self.current_module.use_port = True
          elif self.port_owner != self.current_module:
            raise RuntimeError('Multiple modules attempt to use the port at the same time.')
        self.current_module.array_list.extend(self.array_accessed)
        if p:
          self.current_module = None
        return ans
      return inner
    return wrapper

  @_require_module(True)
  def read(self, onchip_array: ChipArray, onchip_offset: int,
           offchip_array: np.ndarray, offchip_offset: int, num: int):
    '''
    Read a chunk of data from off-chip to on-chip
    '''
    if onchip_array.num < onchip_offset + num:
      raise MemoryError('onchip_array out of bound.')
    # Execute
    self.set_array_mode(onchip_array, ChipArray.READ)
    onchip_array.data[onchip_offset:onchip_offset + num] = \
      offchip_array[offchip_offset:offchip_offset + num]
    self._add_cycles(math.ceil(self.mem_overhead + self.mem_unit * 4 * num))

  @_require_module(True)
  def read_and_partition(self, array_offset_list: list,
                         offchip_array: np.ndarray, offchip_offset: int,
                         num: int, mode: str='block'):
    '''
    Reads an array and partition into several on-chip arrays.
    Requires the same number of cycles as read.
    Args:
      array_offset_list: Each item is (onchip_array, onchip_offset)
      mode: Consider reading 14 items into 3 arrays
        block: [1 2 3 4 5] [6 7 8 9 10] [11 12 13 14 x]
        cycle: [1 4 7 10 13] [2 5 8 11 14] [3 6 9 12 x]
    '''
    n_array = len(array_offset_list)
    num_per_array = math.ceil(num / n_array)
    num_after_padding = num_per_array * n_array
    padding = num_after_padding - num
    in_array = np.concatenate((offchip_array[offchip_offset:offchip_offset + num],
                               np.zeros((padding,))))
    order = 'C' if mode == 'block' else 'F'
    in_array = in_array.reshape((n_array, num_per_array), order=order)

    i = 0
    for onchip_array, onchip_offset in array_offset_list:
      if onchip_array.num < onchip_offset + num_per_array:
        raise MemoryError('onchip_array out of bound.')
      self.set_array_mode(onchip_array, ChipArray.READ)
      onchip_array.data[onchip_offset:onchip_offset + num_per_array] = \
        in_array[i, :]
      i += 1
    self._add_cycles(math.ceil(self.mem_overhead + self.mem_unit * 4 * num))

  @_require_module(True)
  def write(self, onchip_array: ChipArray, onchip_offset: int,
            offchip_array: np.ndarray, offchip_offset: int, num: int):
    '''
    Write a chunk of data from on-chip to off-chip
    '''
    if onchip_array.num < onchip_offset + num:
      raise MemoryError('onchip_array out of bound.')
    # Execute
    self.set_array_mode(onchip_array, ChipArray.WRITE)
    offchip_array[offchip_offset:offchip_offset + num] = \
      onchip_array.data[onchip_offset:onchip_offset + num]
    self._add_cycles(math.ceil(self.mem_overhead + self.mem_unit * 4 * num))

  @_require_module(False)
  def get_item(self, a: ChipArray, index: int) -> float:
    self.set_array_mode(a, ChipArray.COMPUTE)
    self._add_cycles(self.index_cycle)
    return a.data[index]

  @_require_module(False)
  def compute(self, value: float, name: str):
    if not name in self.dsp_list:
      if len(self.dsp_list) == self.dsp_num:
        raise RuntimeError('Number of DSPs exceeded.')
      self.dsp_list.append(name)
    self._add_cycles(self.dsp_cycle)
    return value

  @_require_module(False)
  def array_write(self, dst_array: ChipArray, dst_offset: int, value: float):
    self.set_array_mode(dst_array, ChipArray.COMPUTE)
    dst_array.data[dst_offset] = value
    self._add_cycles(self.index_cycle)
    return value

  def module(self, func=None, name: str=''):
    # self._check_onchip(array_list)
    module = ChipModule(self, name, func)
    self.module_list.append(module)
    return module

  def enter_module(self, module):
    self.current_module = module

  def exit_module(self):
    self.module_history.append((self.current_module, self.current_module.cycles))
    # print(self.current_module.name)
    # print(self.current_module.cycles)
    self.current_module = None

  def unrolled_loop(self, n: int, f):
    '''
    Run the loop f for n times
    f takes one argument idx: the loop index
    Fully unrolled
    Must reside in a module
    '''
    assert(self.current_module is not None)
    if n <= 0:
      return
    old_cycles = self.current_module.cycles
    # print('old_cycles = {}'.format(old_cycles))
    max_cycles = old_cycles
    for i in range(n):
      f(idx=i)
      max_cycles = max(max_cycles, self.current_module.cycles)
      # print('max_cycles = {}'.format(max_cycles))
      self.current_module.cycles = old_cycles
    self.current_module.cycles = max_cycles

  def join(self):
    '''
    Wait for all modules to finish
    Flips all flip-flops
    Resets all ports
    '''

    if len(self.module_group_stack) == 0:
      # Not in a module group
      # Recover the port
      self.port_owner = None

      # Recover the arrays
      for a in self.array_list:
        a.mode = ChipArray.UNUSED
        a.module_owner = None

      # Flip the flip-flops
      for ff in self.ff_list:
        ff.flip()

      # Recover the modules
      for m in self.module_list:
        m.use_port = False
        m.array_list = []
        m.cycles = 0

      # Update cycles
      l0 = 0 if len(self.join_index) == 0 else self.join_index[-1]
      l1 = len(self.module_history)
      self.join_index.append(l1)
      cycles = 0
      for i in range(l0, l1):
        cycles = max(cycles, self.module_history[i][1])
      self.cycles += cycles
    else:
      # #1: In a module group, mg = innermost module group
      mg = self.module_group_stack[-1]

      # Recover the port, and write mg.use_port
      if self.port_owner is not None:
        mg.use_port = True
      self.port_owner = None

      # Check newly visited modules (mg.mi to l are new modules)
      # Also recover modules, and update mg.join_index
      l = len(self.module_history)
      array_accessed = []
      k = mg.mi
      while k < l:
        m = self.module_history[k]
        mg.module_history.append(m)  # insert into mg.module_history
        m = m[0] # m = the module
        k += 1
        for a in m.array_list:
          if a not in array_accessed:
            array_accessed.append(a)
        # Recover module m
        m.use_port = False
        m.array_list = []
        m.cycles = 0
      mg.join_index.append(len(mg.module_history))  # update join_index

      # Update cycles (only mg.cycles, not self.cycles)
      cycles = 0
      for i in range(mg.mi, l):
        cycles = max(cycles, self.module_history[i][1])
      mg.cycles += cycles
      mg.mi = l

      # Recover arrays and update mg.array_list
      for a in array_accessed:
        # Recover array a
        a.mode = ChipArray.UNUSED
        a.module_owner = None
        # Add a to mg.array_list
        if a not in mg.array_list:
          mg.array_list.append(a)

      # Flip the flip-flops (#1: might have problems here)
      for ff in self.ff_list:
        ff.flip()

  def module_group(self, name: str):
    return ChipModuleGroup(self, name)

  def push_module_group(self, mg):
    self.module_group_stack.append(mg)

  def pop_module_group(self, mg):
    assert self.module_group_stack[-1] == mg
    self.module_group_stack = self.module_group_stack[:-1]

  def read_cycles(self):
    # #1: must use this function to access self.cycles
    assert self.join_index[-1] == len(self.module_history) # cannot read before join
    return self.cycles

  def reset(self):
    for m in self.module_list:
      m.reset()
    self.module_history = []
    self.join_index = []
    self.cycles = 0

  def print_summary(self):
    Chip._print_summary(self, 0 if self.verbose else -1)

  @staticmethod
  def _print_summary(mg, n: int):
    # #1
    # mg can be a chip or a module group
    # n = -1 then not verbose
    l0 = 0
    for l1 in mg.join_index:
      cycles = 0
      for i in range(l0, l1):
        print('{}Module: {}\tCycles: {}'.format(' ' * n * 4,
                                                mg.module_history[i][0].name, mg.module_history[i][1]))
        cycles = max(cycles, mg.module_history[i][1])
        if n >= 0 and type(mg.module_history[i][0]) is ChipModuleGroup:
          Chip._print_summary(mg.module_history[i][0], n + 1)
      print('{}+ Block Cycles: {}'.format(' ' * n * 4, cycles))
      print('{}=================='.format(' ' * n * 4))
      l0 = l1
    if n == 0 or n == -1:
      # assert mg is chip
      print('Total Cycles: {}'.format(mg.read_cycles()))


class ChipModule(object):
  def __init__(self, chip: Chip, name: str, func):
    self.chip = chip
    self.name = name
    self.cycles = 0
    self.func = func
    self.reset_func = None
    self.use_port = False
    self.array_list = []

  def set_func(self, func):
    self.func = func

  def __call__(self, *args, **kwargs):
    assert (self.chip.current_module is None)
    self.chip.enter_module(self)
    ans = self.func(*args, **kwargs)
    self.chip.exit_module()
    return ans

  def set_reset(self, func):
    self.reset_func = func

  def reset(self):
    if self.reset_func is not None:
      self.reset_func()


# #1: Module group
class ChipModuleGroup(object):
  def __init__(self, chip: Chip, name: str):
    self.chip = chip
    self.name = name
    self.cycles = 0
    self.use_port = False
    self.module_history = []
    self.array_list = []
    self.join_index = []
    # All modules between mstart and mend are in this group
    self.mstart = 0   # The # of modules when this group starts
    self.mi = 0

  def __enter__(self):
    self.mstart = len(self.chip.module_history)
    self.mi = self.mstart
    self.chip.push_module_group(self)

  def __exit__(self, exc_type, exc_val, exc_tb):
    # Check if joined correctly
    assert self.join_index[-1] == len(self.chip.module_history) - self.mstart

    # Recover arrays
    for a in self.array_list:
      a.module_owner = self

    # Recover port
    if self.use_port:
      assert self.chip.port_owner is None
      self.chip.port_owner = self

    # Merge modules into the group
    self.chip.module_history = self.chip.module_history[:self.mstart]
    self.chip.module_history.append((self, self.cycles))

    # Pop
    self.chip.pop_module_group(self)
