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
    self.data = np.random.randn(num)
    self.mode = ChipArray.UNUSED
    self.module_owner = None

  def __getitem__(self, index):
    raise RuntimeError('Cannot access an array item directly. Must use Chip.get_item()')


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

    # For analysis
    self.verbose = False
    self.module_history = []
    self.join_index = []

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
  def compute(self, value: float):
    self._add_cycles(self.dsp_cycle)
    return value

  @_require_module(False)
  def array_write(self, dst_array: ChipArray, dst_offset: int, value: float):
    self.set_array_mode(dst_array, ChipArray.COMPUTE)
    dst_array.data[dst_offset] = value
    self._add_cycles(self.index_cycle)
    return value

  def module(self, func, name: str=''):
    # self._check_onchip(array_list)
    module = ChipModule(self, name, func)
    self.module_list.append(module)
    return module

  def enter_module(self, module):
    self.current_module = module

  def exit_module(self):
    self.current_module = None

  def join(self):
    # Recover the port and arrays
    self.port_owner = None
    for a in self.array_list:
      a.mode = ChipArray.UNUSED
      a.module_owner = None
    for m in self.module_list:
      # Debug
      # print('Module: {}'.format(m.name))
      # print('Use port: {}'.format(m.use_port))
      # for a in m.array_list:
      #   print(a.name)

      m.use_port = False
      m.array_list = []
      m.cycles = 0
    # Update cycles
    # TODO
    self.join_index.append(len(self.module_history))



class ChipModule(object):
  def __init__(self, chip: Chip, name: str, func):
    self.chip = chip
    self.name = name
    self.cycles = 0
    self.func = func
    self.use_port = False
    self.array_list = []

  def __call__(self, *args, **kwargs):
    assert (self.chip.current_module is None)
    self.chip.enter_module(self)
    ans = self.func(*args, **kwargs)
    self.chip.exit_module()
    return ans
