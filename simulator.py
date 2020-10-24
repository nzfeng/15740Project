class Chip(object):
  def __init__(self, dsp_num, dsp_cycle, mem_size, mem_overhead,
               mem_unit):
    self.dsp_num = dsp_num
    self.dsp_cycle = dsp_cycle
    self.mem_size = mem_size
    self.mem_overhead = mem_overhead
    self.mem_unit = mem_unit
