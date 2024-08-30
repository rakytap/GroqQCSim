import groq.api as g
import groq.api.nn as nn
import groq.runner.tsp as tsp;
from groq.api import instruction as inst
import numpy as np
from QCsim_IOP import Gate_IOP


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

	

	program = Gate_IOP(pgm_pkg, mxm_permutation = mxm_permutation)
	# Defines a program to upload the inputs
	program.upload_program()
	program.download_state_program()
	program.split_gates_program()

	# program to perform gate_S operation
	program.gate_S_program()

	# program to perform gate_SS operation
	program.gate_SS_program()

	# program to perform gate_SL operation
	#program.gate_SL_program()
	
	iops = pgm_pkg.assemble()

	return iops



def invoke(device, iop, pgm_num, ep, tensors):
	"""Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""

	pgm = iop[pgm_num]
	#ep = pgm.entry_points[ep_num]
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


def run(iop_file, input_real, input_imag, target_qbits, control_qbits, gate_kernels_real, gate_kernels_imag, mxm_permutation = True):
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
	device.load(iop[3], unsafe_keep_entry_points=True)
	device.load(iop[4], unsafe_keep_entry_points=True)
	#device.load(iop[5], unsafe_keep_entry_points=True)
	'''
	if target_qbit >= small_qbit_num_limit :
		raise ValueError( "the target qubit in this program should be less than 8" )

	'''
	
	# maps for mem_gather to select the correct state_1 indices according to the target qubit
	target_qbit_selector = np.zeros( (gate_count,320,), dtype=np.uint8 )
	for gate_idx in range(gate_count):
		target_qbit_selector[gate_idx, 0:320:16] = target_qbits[gate_idx]
		
	state_1_selector = target_qbit_selector.copy()#np.zeros( (2,320,), dtype=np.uint8 )
	state_0_selector = target_qbit_selector.copy()#np.zeros( (2,320,), dtype=np.uint8 )	
	
	control_state_1_selector = np.zeros( (gate_count,320,), dtype=np.uint8 )
	for gate_idx in range(gate_count):
		
		control_qbit = control_qbits[gate_idx]

		if ( control_qbit < small_qbit_num_limit and control_qbit >= 0 and control_qbit<8) :
			control_state_1_selector[gate_idx,0:320:16] = control_qbit
		else :
			control_state_1_selector[gate_idx,0:320:16] = 0  # no control qubit is used when filetring the gate kernels
	control_state_0_selector = control_state_1_selector
	
	
	# create memory maps to access the correct metadata on the chip to perform the gate operation on the chosen target_qubit
	#if mxm_permutation :
	#in case the MXM is used, the individual permute mtrices will occupy 20 memory addresses
	mxm_permutation_selector = target_qbit_selector*20
	#else:  ##### !!! gate SL needs the permutor !!! #####
	
	# in case the permutor is used for permutation
	permutation_selector = target_qbit_selector	


	# control qubit distributor maps for SL gates
	distributor_map_np = np.zeros( (gate_count, 320,), dtype=np.uint8 )
	for gate_idx in range(gate_count):
	
		control_qbit = control_qbits[gate_idx
]
		if control_qbit > 8 and control_qbit <= 16:
			distributor_map_np[gate_idx, 0:320:16] = 2*(control_qbit - small_qbit_num_limit - 1)
			distributor_map_np[gate_idx, 1:320:16] = 2*(control_qbit - small_qbit_num_limit - 1) + 1
			

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

	



	# run the upload program
	entry_point = iop["upload"]["mono"]
	pgm_1_output = invoke(device, iop, 0, entry_point, {"State_input_real": input_real, "State_input_imag": input_imag, "mxm_permutation_selector": mxm_permutation_selector, "permutation_selector": permutation_selector, "gate_kernels_real_packed": gate_kernels_real_packed, "gate_kernels_imag_packed": gate_kernels_imag_packed, "state_1_selector": state_1_selector, "state_0_selector": state_0_selector, "control_state_1_selector": control_state_1_selector, "control_state_0_selector": control_state_0_selector, "control_qubit_distributor_maps": distributor_map_np})

	# run the gate split program
	index_of_split_program = 2
	entry_point = iop["split_gates"]["compute"] #4*index_of_split_program + 1
	pgm_2_output = invoke(device, iop, index_of_split_program, entry_point, {})

	#print( pgm_2_output )

	for gate_idx in range(gate_count):

		target_qbit  = target_qbits[gate_idx]
		control_qbit = control_qbits[gate_idx]
	
		# run the gate program
		if target_qbit == control_qbit:
			error("QCsim::run: The control qubit and target qubit should be different")

		elif target_qbit >= qbit_num:
			error("QCsim::run: The target qubit should be less than teh number of qubits")

		elif control_qbit >= qbit_num:
			error("QCsim::run: The control qubit should be less than teh number of qubits")

		elif target_qbit >= 0 and target_qbit < 8 and control_qbit == -1: 
			# pogram perfroming the Gate_S operation
			index_of_gate_program = 3
			entry_point = iop["gate_S"]["compute"]
			gate_output = invoke(device, iop, index_of_gate_program, entry_point, {})

		elif target_qbit >= 0 and target_qbit < 8 and control_qbit < 8 and control_qbit >= 0: 

			# pogram perfroming the Gate_SS operation
			index_of_gate_program = 4
			entry_point = iop["gate_SS"]["compute"]
			gate_output = invoke(device, iop, index_of_gate_program, entry_point, {})	

		elif target_qbit >= 0 and target_qbit < 8 and control_qbit > 8 and control_qbit <= 16:

			# pogram perfroming the Gate_SL operation
			index_of_gate_program = 5
			entry_point = iop["gate_SL"]["compute"]
			gate_output = invoke(device, iop, index_of_gate_program, entry_point, {})
			#print( gate_output["test"] )	
			#print( gate_output["test"].dtype )	
			#print( gate_output["test"].shape )		

		else:

			error("QCsim::run: Unimplemented case wit target qubit and control qubit.")
	
	

	# download the transformed state vector
	index_of_download_program = 1
	
	entry_point = iop["download_state_vector"]["output"]
	download_program_output = invoke(device, iop, index_of_download_program, entry_point, {})

	Psi_transformed_real = download_program_output["State_transformed_real"]
	Psi_transformed_real = np.reshape( Psi_transformed_real, (Psi_transformed_real.size,) )
	
	Psi_transformed_imag = download_program_output["State_transformed_imag"]
	Psi_transformed_imag = np.reshape( Psi_transformed_imag, (Psi_transformed_imag.size,) )	
	
	return Psi_transformed_real, Psi_transformed_imag



def main( State_orig_real_float32, State_orig_imag_float32, target_qbits, control_qbits, gate_kernels, mxm_permutation = True ):
	"""Compiles and runs the example programs."""
	
	if qbit_num is None:
		raise "qbit_num must be set with method initialize_qbit_num( qbit_num ) with a value greater or equal to 8"

	iop_files = compile( mxm_permutation )
	print(f"Program compiled to IOP file: {iop_files}")

	import time

	start_time = time.time()
	transformed_state = run(iop_files[0], State_orig_real_float32, State_orig_imag_float32, target_qbits, control_qbits, np.real(gate_kernels), np.imag(gate_kernels), mxm_permutation )

	print("time elapsed with groq: ", time.time() - start_time )
	
	return transformed_state




if __name__ == "__main__":
    main()
    
