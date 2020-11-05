# Simulator for EIE, an accelerator which performs sparse matrix-vector 
# multiplication.

import numpy as np
from scipy.sparse import random
import argparse

from simulator import Chip

MV_dim = 4 # size of the matrix
sparsity = 0.5 # sparsity of W and a, as a fraction of nonzero elements
num_PE = 1 # number of PEs (processing elements)
queue_depth = 0 # depth of activation queue in each PE as a power of 2
num_layers = 1 # number of FC layers


def isCorrect(W, a, b):
	'''
	Use naive multiplication to confirm if the results of EIE are indeed 
	correct, up to floating-point error.

	Args:
	    W = weight matrix
	    a = input vector
	    b = output vector computed by EIE

	Returns:
	    True if multiplication checks out, False otherwise
	'''
	epsilon = 1e-5
	[rows, cols] = np.shape(W)
	valid = np.zeros(rows)
	for i in range(rows):
		for j in range(cols):
			valid[i] += W[i,j]*a[j]
		if (abs(valid[i] - b[i]) > epsilon):
			return False
	return True


def matrixToCSC(M : np.ndarray):
	'''
	Convert a matrix M into the compressed sparse column (CSC) format described
	in the paper. The paper uses 4 bits as the size of each element, but in our 
	simulator we use floats throughout, so we won't worry about extra 0's in 
	our encoding.

	Args:
		Matrix to be compressed. Could also be just a single column vector.

	Returns:
		v = array of nonzero elements
		z = array representing relative row index, i.e. # of zeros before the
			corresponding element in v (in that column)
		p = array of column pointers, i.e. index in v where each new column
			begins. Last element points one beyond the last element.
	'''
	(rows, cols) = np.shape(M)
	v = []
	z = [] # changed this to be row index instead
	p = []
	for j in range(cols):
		num_zeros = 0
		flag = False # if first nonzero element in a column has been located
		for i in range(rows):
			elem = M[i,j]
			if (elem != 0):
				if (not flag): p.append(len(v))
				v.append(elem)
				z.append(i)
				# z.append(num_zeros)
				num_zeros = 0
				flag = True
			else:
				num_zeros += 1
		if (not flag): p.append(len(v))

	p.append(len(v))
	# sanity checking
	if (len(v) != len(z)): print("len(v) != len(z_")
	if (len(p) != cols + 1): print("len(p) incorrect")

	# Convert to numpy arrays.
	v = np.asarray(v)
	z = np.asarray(z)
	p = np.asarray(p)

	return [v, z, p]



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dsp_num', '-D', type=int,
		help='Number of DSPs')
	parser.add_argument('--dsp_cycle', '-C', type=int,
		help='Cycles a DSP takes for one flop')
	parser.add_argument('--index_cycle', '-X', type=int,
		help='Cycles for array indexing')
	parser.add_argument('--mem_size', '-M', type=int,
		help='Size of on-chip memory (in bytes)')
	parser.add_argument('--mem_overhead', '-A', type=int,
		help='Overhead of read-write port')
	parser.add_argument('--mem_unit', '-B', type=float,
		help='Cycles for reading 1 byte from the port')
	parser.add_argument('--matrix_size', '-MV_dim', type=float,
		help='Maximum dimension of matrix/vector')
	parser.add_argument('--sparsity', '-density', type=float,
		help='Density of the weight matrix W and initial input vector')
	parser.add_argument('--PE_num', '-PE_num', type=float,
		help='Number of PEs')

	args = parser.parse_args()
	MV_dim = int(args.matrix_size)
	sparsity = args.sparsity
	num_PE = int(args.PE_num)

	chip = Chip(args.dsp_num, args.dsp_cycle, args.index_cycle,
		args.mem_size, args.mem_overhead, args.mem_unit)

	# Create random sparse matrix W, and sparse vector a.
	# Portions of W are put into CSC format by CPU.
	W = random(MV_dim, MV_dim, density=sparsity, dtype=float)
	W = W.A
	a = random(MV_dim, 1, density=sparsity, dtype=float)
	a = a.A
	b = np.zeros((MV_dim, 1), dtype=float) # will hold final computed result
	#print(W)
	X = np.zeros(num_PE) # encodes location of nonzero elements in input vector

	# A list of v, z, p vectors (CSC representations) for each portion of W in 
	# each PE.
	VZP = [] # = [[v1, z1, p1], [v2, z2, p2],..., [vN, zN, pN]]
	# We assume that the number of PEs evenly divides the number of rows.
	num_rows = MV_dim//num_PE # number of rows of W sent to each PE
	for k in range(num_PE):
		# Extract rows of W for which (i mod num_PE) = k.
		W_section = np.empty(shape=(num_rows,MV_dim)) 
		for i in range(num_rows):
			W_section[i:i+1, :] = W[k+i*num_PE:k+i*num_PE+1, :]

		VZP.append(matrixToCSC(W_section))

	# Allocate space for the portions of W, b on each PE; plus the activation
	# queue.
	len_V = 0
	for k in range(num_PE):
		len_V += len(VZP[k][0])
	V = chip.array(len_V, "V")
	Z = chip.array(len_V, "Z")
	P = chip.array((MV_dim+1)*num_PE, "P")
	offsets = chip.array(num_PE, "offsets") # offsets delimiting where each PE begins
	A = chip.array(MV_dim, "A")
	B = chip.array(MV_dim, "B")
	AJ = chip.array(1, "AJ") # activation index
	AV = chip.array(1, "AQ") # activation value

	def f_readInput():
		'''
		Read arrays onto chip.
		'''
		offset_vec = np.zeros(num_PE)
		onchip_offset = 0
		for k in range(num_PE):
			num = len(VZP[k][0])
			chip.read(V, onchip_offset, VZP[k][0], 0, num)
			chip.read(Z, onchip_offset, VZP[k][1], 0, num)
			offset_vec[k] = onchip_offset
			onchip_offset += num
			chip.read(P, k*(MV_dim+1), VZP[k][2], 0, MV_dim+1)
			# Read input vector. Only have to do this for the 1st layer.
			# (consider breaking this out into another function)
			chip.read(A, num_rows*k, [a[i] for i in range(k, MV_dim, num_PE)], 0, num_rows)
			# Initialize output array into which results will be accumulated.
			chip.read(B, 0, np.zeros(MV_dim), 0, MV_dim)
		chip.read(offsets, 0, offset_vec, 0, num_PE)


	def f_LNZD(j):
		'''
		Given a vector, return the value and index of the leading nonzero element.
		This function is performed by each PE for the parallel determination
		of nonzero elements in a.

		Args:
			j = index in each PE's portion of a that we're checking. 
				0 <= j < num_rows

		Returns:
			Updates X register, which we assume takes 0 cycles to read/write from.
			X[i] = 1 if the jth element in the portion of a owned by the ith PE is
			nonzero (0 otherwise.)
		'''

		# X has length = num_PE = num_LNZD.
		for k in range(num_PE): # to be unrolled
			if (chip.get_item(A, k*num_rows + j) != 0):
				X[k] = 1

	def f_broadcast(j):

		for k in range(num_PE):
			if (X[k] > 0):
				chip.array_write(AJ, 0, j*num_PE+k) # index in the original
				chip.array_write(AV, 0, chip.get_item(A, k*num_rows + j))
				X[k] = 0
				return True	
		# X should be all zero again
		return False


	def f_processingElement():
		'''
		Simulate an array of PEs running in parallel.
		'''

		# Get the next element of a from the activation queue.
		a_j = chip.get_item(AV, 0)
		j = (chip.get_item(AJ, 0)).astype(int)

		for k in range(num_PE): # to be unrolled
			# Multiply every element of W in column j with a.
			offset = (chip.get_item(offsets, k)).astype(int)
			col_start = (chip.get_item(P, (MV_dim+1)*k+j)).astype(int)
			col_end = (chip.get_item(P, (MV_dim+1)*k+(j+1))).astype(int)
			for i in range(col_start, col_end):
				w_ij = chip.get_item(V, offset+i)
				z = (chip.get_item(Z, offset+i)).astype(int) # relative row index
				b_i = chip.get_item(B, k*num_rows+z)
				add = chip.compute(w_ij*a_j)
				res = chip.compute(b_i + add)
				chip.array_write(B, k*num_rows+z, res)


	def f_writeOutput():
	    # If we assume each PE grabs a contiguous block of rows: saves on 
	    # indexing, but sparse matrix arising from a pruned DNN might have a 
	    # more regular sparsity pattern (i.e. localized blocks of zeros.) In 
	    # that case, PEs will have more uneven balance of work.
	    for k in range(num_PE):
		    for i in range(num_rows):
		    	chip.write(B, k*num_rows+i, b[num_PE*i+k], 0, 1) 


	readInput = chip.module(f_readInput, 'read')
	LNZD = chip.module(f_LNZD, 'LNZD')
	broadcast = chip.module(f_broadcast, 'broadcast')
	processingElements = chip.module(f_processingElement, 'PE')
	writeOutput = chip.module(f_writeOutput, 'write')

	readInput()
	chip.join()
	for m in range(num_rows):
		#r = True
		LNZD(m)
		chip.join()
		while (broadcast(m)):
			chip.join()
			processingElements()
			chip.join()
	writeOutput()
	chip.join()

	# Make sure EIE performed SPMV correctly.
	#print(b)
	#print(np.matmul(W,a))
	print("SPMV %s" %isCorrect(W, a, b))

if __name__ == '__main__':
  main()