import groq.api as g
import groq.api.nn as nn
import groq.runner.tsp as tsp;
from groq.api import instruction as inst
import numpy as np
from gate import Gate

import os
import shutil
from typing import List
import utils
from memory_layouts import *

try:
    import groq.runtime.driver as runtime
except ImportError:
    # raise ModuleNotFoundError("groq.runtime")
    print('Error: ModuleNotFoundError("groq.runtime")')

import time
print("Python packages imported successfully")


#Note: MEM slices 16, 20, 24, and 28 on both the east and west hemispheres, and slice 38 on the west hemisphere are reserved for system use.


def compile(mxm_permutation = True) -> List[str]:
	"""Compiles a program package with 2 programs.

	Return: (List[str]): A list of IOP files.
	"""
	output_dir = "./build_iop"
	shutil.rmtree(output_dir, ignore_errors=True)
	pgm_pkg = g.ProgramPackage("QCsim_multi_program", output_dir)

	

	program = Gate(pgm_pkg, mxm_permutation = mxm_permutation)
	# Defines a program to upload the inputs
	program.upload_program()
	program.split_gates_program()
	# program to perform gate operation in a target qubit less than small_qbit_num_limit
	program.gate_program()
	
	iops = pgm_pkg.assemble()

	return iops



def invoke(device, iop, pgm_num, ep_num, tensors):
	"""Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""

	pgm = iop[pgm_num]
	ep = pgm.entry_points[ep_num]
	input_buffer = runtime.BufferArray(ep.input, 1)[0]
	output_buffer = runtime.BufferArray(ep.output, 1)[0]

	if ep.input.tensors:
		for input_tensor in ep.input.tensors:
			if input_tensor.name not in tensors:
				raise ValueError(f"Missing input tensor named {input_tensor.name}")
			input_tensor.from_host(tensors[input_tensor.name], input_buffer)

	device.invoke(input_buffer, output_buffer)
	outs = {}

	if ep.output.tensors:
		for output_tensor in ep.output.tensors:
			result_tensor = output_tensor.allocate_numpy_array()
			output_tensor.to_host(output_buffer, result_tensor)
			outs[output_tensor.name] = result_tensor
	return outs


def run(iop_file, input_real, input_imag, target_qbit, gate_kernels_real, gate_kernels_imag, mxm_permutation = True):
	"""
    This function interacts with the device driver at a lower level to show
    the control of loading 2 programs and invoking each program through
    program entry points. A higher level abstraction will be made available
    to load and invoke multiple programs in a future release.
	"""

	np.set_printoptions(linewidth=1000, threshold=10000)

	if not os.path.exists(iop_file):
		raise Exception(f"IOP file does not exist: {iop_file}")

	print(f"Running programs from {iop_file}")



	iop = runtime.IOProgram(iop_file)
	device = runtime.devices[0]
	device.open()
	device.load(iop[0], unsafe_keep_entry_points=True)
	device.load(iop[1], unsafe_keep_entry_points=True)
	device.load(iop[2], unsafe_keep_entry_points=True)
	
	if target_qbit >= small_qbit_num_limit :
		raise ValueError( "the target qubit in this program should be less than 8" )
	
	
	# maps for mem_gather to select the correct state_1 indices according to the target qubit
	target_qbit_selector = np.zeros( (320,), dtype=np.uint8 )
	target_qbit_selector[0:320:16] = target_qbit
		
	state_1_selector = target_qbit_selector#np.zeros( (2,320,), dtype=np.uint8 )
	state_0_selector = target_qbit_selector#np.zeros( (2,320,), dtype=np.uint8 )	
	
	
	# create memory maps to access the correct metadata on the chip to perform the gate operation on the chosen target_qubit
	if mxm_permutation :
		state_permute_map_selector = None
		
		#map to choose the correct permutation matrix to reorder the state vector
		mxm_permutation_matrix_selector = None
	
	
	else:
		# map for mem_gather to select the permutation map for the given target qubit. Th epermutor is used only for target qubits smaller than small_qbit_num_limit
		state_permute_map_selector = target_qbit_selector
			
		mxm_permutation_matrix_selector = None
			


	# generate packed gate kernel data
	gate_kernels_real_packed = np.zeros( (320,), dtype=np.float32 )
	gate_kernels_imag_packed = np.zeros( (320,), dtype=np.float32 )

	gate_kernels_real_shape = gate_kernels_real.shape
	gate_kernels_imag_shape = gate_kernels_imag.shape

	if gate_kernels_real_shape != gate_kernels_imag_shape:
		raise ValueError( "the shape of the real part and the imaginary part of gate kernels should match" )


	if gate_kernels_real_shape[1] != 2 or gate_kernels_real_shape[2] != 2:
		raise ValueError( "gate kernels should be of size 2x2" )


	gate_num = gate_kernels_real_shape[0]

	gate_kernels_real_packed = np.ascontiguousarray( gate_kernels_real.reshape( (-1,) ) )
	gate_kernels_imag_packed = np.ascontiguousarray( gate_kernels_imag.reshape( (-1,) ) )


	# run the first program
	if mxm_permutation :
		pgm_1_output = invoke(device, iop, 0, 0, {"State_input_real": input_real, "State_input_imag": input_imag, "State_input_real_copy": input_real, "State_input_imag_copy": input_imag, "gate_kernels_real_packed": gate_kernels_real_packed, "gate_kernels_imag_packed": gate_kernels_imag_packed, "state_1_selector": state_1_selector, "state_0_selector": state_0_selector})
		
	else: 
		pgm_1_output = invoke(device, iop, 0, 0, {"State_input_real": input_real, "State_input_imag": input_imag, "State_input_real_copy": input_real, "State_input_imag_copy": input_imag, "state_permute_map_selector": state_permute_map_selector, "gate_kernels_real_packed": gate_kernels_real_packed, "gate_kernels_imag_packed": gate_kernels_imag_packed, "state_1_selector": state_1_selector, "state_0_selector": state_0_selector})
	
	
	# run the gate split program
	index_of_split_program = 1
	pgm_2_output = invoke(device, iop, index_of_split_program, 0, {})
	#print( pgm_2_output.keys() )
	'''
	print( pgm_2_output["gate_kernel_broadcasted_real_0_00"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_0_00"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_1_00"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_1_00"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_0_01"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_0_01"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_1_01"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_1_01"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_0_10"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_0_10"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_1_10"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_1_10"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_0_11"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_0_11"] )
	print(' ')
	print( pgm_2_output["gate_kernel_broadcasted_real_1_11"] )
	print( pgm_2_output["gate_kernel_broadcasted_imag_1_11"] )
	print(' ')
	print( pgm_2_output["gate_kernel_real_0"] )
	print( pgm_2_output["gate_kernel_imag_0"] )
	'''

	# run the second program
	index_of_gate_program = 2
	pgm_3_output = invoke(device, iop, index_of_gate_program, 0, {})
	#print(pgm_3_output["tmp_imag_mt"] == pgm_3_output["tmp_imag_mt2"])
	#print(pgm_3_output["tmp_real_mt"] == pgm_3_output["tmp_real_mt2"])
	#print(pgm_3_output["gate_first_column_real"])
	#print(pgm_3_output["gate_first_column_imag"])
	#print(pgm_3_output["Psi_transformed_real"])
	#print(pgm_3_output["Psi_transformed_imag"])

	Psi_transformed_real = pgm_3_output[f"Psi_transformed_real"]
	Psi_transformed_real = np.reshape( Psi_transformed_real, (Psi_transformed_real.size,) )
	
	Psi_transformed_imag = pgm_3_output[f"Psi_transformed_imag"]
	Psi_transformed_imag = np.reshape( Psi_transformed_imag, (Psi_transformed_imag.size,) )	
	return Psi_transformed_real, Psi_transformed_imag



def main( State_orig_real_float32, State_orig_imag_float32, target_qbit, gate_kernels, mxm_permutation = True ):
	"""Compiles and runs the example programs."""
	
	if qbit_num is None:
		raise "qbit_num must be set with method initialize_qbit_num( qbit_num ) with a value greater or equal to 8"

	iop_files = compile( mxm_permutation )
	print(f"Program compiled to IOP file: {iop_files}")

	import time

	start_time = time.time()
	transformed_state = run(iop_files[0], State_orig_real_float32, State_orig_imag_float32, target_qbit, np.real(gate_kernels), np.imag(gate_kernels), mxm_permutation )

	print("time elapsed with groq: ", time.time() - start_time )
	
	return transformed_state




if __name__ == "__main__":
    main()
    
