import numpy as np


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

  def __getitem__(self, index) -> float:
    return self.data[index]


class Chip(object):
  def __init__(self, dsp_num: int, dsp_cycle: int, mem_size: int,
               mem_overhead: int, mem_unit: float):
    self.dsp_num = dsp_num
    self.dsp_cycle = dsp_cycle
    self.mem_num = mem_size // 4  # Float is 4 bytes
    self.mem_overhead = mem_overhead
    self.mem_unit = mem_unit

    self.array_list = []
    self.free_mem_num = self.mem_num
    self.port_owner = None  # Port between on-chip and off-chip

    self.current_module = None
    self.module_list = []
    self.join_index = []
    self.auto_module_mode = False

    self.array_accessed = []

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

  def register(self, init_value=0.0) -> ChipArray:
    a = ChipArray(1, '')
    a.data[0] = init_value
    return a

  def _is_onchip(self, onchip_array: ChipArray) -> bool:
    idx = -1
    l = len(self.array_list)
    for i in range(l):
      if self.array_list[i] == onchip_array:
        idx = i
        break
    return idx > -1

  def check_onchip(self, array_list: list):
    for a in array_list:
      if not self._is_onchip(a):
        raise MemoryError('Array {} is not on this chip.'.format(a.name))

  def set_array_mode(self, a: ChipArray, mode: int):
    assert(a not in self.array_accessed)
    if self.auto_module_mode and a.mode != ChipArray.UNUSED:
      raise RuntimeError('Array {} belongs to multiple modules.'.format(a.name))
    if a.mode not in [mode, ChipArray.UNUSED]:
      raise RuntimeError('Array {} used for multiple purposes.'.format(a.name))
    a.mode = mode
    self.array_accessed.append(a)

  def _require_module(use_port: bool):
    def wrapper(func):
      def inner(self, *args, **kwargs):
        self.array_accessed = []
        if self.current_module is None:
          # Construct a module for func. func needs to return the array_list
          self.auto_module_mode = True
          l = len(self.module_list)
          module = self.module(use_port, [], 'auto_module_{}'.format(l))
          self.current_module = module
          func(self, *args, **kwargs)
          self.check_onchip(self.array_accessed)

          module.array_list = self.array_accessed
          self.current_module = None
          self.auto_module_mode = False
        else:
          # Check the port
          if use_port and self.port_owner != self.current_module:
            raise RuntimeError('Attempt to use the port in a module that does not own the port.')

          # Execute and check whether all arrays belong to the module
          func(self, *args, **kwargs)
          for a in self.array_accessed:
            if a not in self.current_module.array_list:
              raise RuntimeError('Array {} is not in the current module.'.format(a.name))
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

  @_require_module(False)
  def compute(self, dst: ChipArray, value: float):
    dst.data[0] = value
    return value

  @_require_module(False)
  def compute_array(self, dst_array: ChipArray, dst_offset: int, value: float):
    self.set_array_mode(dst_array, ChipArray.COMPUTE)
    dst_array.data[dst_offset] = value
    return value

  def module(self, use_port: bool, array_list: list, name: str=''):
    module = ChipModule(self, use_port, array_list, name)
    self.module_list.append(module)
    return module

  def join(self):
    self.join_index.append(len(self.module_list))
    # Recover the port and arrays
    self.port_owner = None
    for a in self.array_list:
      a.mode = ChipArray.UNUSED
    # Update cycles
    # TODO


class ChipModule(object):
  def __init__(self, chip: Chip, use_port: bool, array_list: list, name: str):
    self.chip = chip
    self.name = name
    if use_port:
      if self.chip.port_owner is not None:
        raise RuntimeError('There is only one port between on-chip and off-chip.')
      self.chip.port_owner = self

    self.array_list = array_list
    self.chip.check_onchip(array_list)
    for a in self.array_list:
      if a.mode != ChipArray.UNUSED:
        raise RuntimeError('Array {} belongs to multiple modules.'.format(a.name))


  def __enter__(self):
    assert(self.chip.current_module is None)
    self.chip.current_module = self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.chip.current_module = None
