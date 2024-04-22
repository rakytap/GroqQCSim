from upload import Upload
from memory_layouts import *
import groq.api as g
import numpy as np
from groq.api import instruction as inst

class Split_gates(Upload):
	def __init__(self, pgm_pkg):
		super().__init__(pgm_pkg)
		self.gate_kernels_broadcasted_real_dict = {}
		self.gate_kernels_broadcasted_imag_dict = {}
		self.gate_kernels_broadcasted_real_copy_dict = {}
		self.gate_kernels_broadcasted_imag_copy_dict = {}		

	def split_gates_program(self):
		name = "split_gates"
		with self.pgm_pkg.create_program_context(name) as pgm2:

			# create reusable permutor requests
			permutor_requests = []
			permutor_requests.append(  g.tensor.create_permutor_request( [0], 1 ) ) # WEST
			permutor_requests.append(  g.tensor.create_permutor_request( [1], 1 ) ) # EAST
			
			# create reusable distributor requests
			distributor_requests = []
			distributor_requests.append(  g.tensor.create_distributor_request( [0], 1 ) ) # WEST
			distributor_requests.append(  g.tensor.create_distributor_request( [4], 1 ) ) # EAST		

			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			gate_kernels_real_packed_shared = g.shared_memory_tensor(self.gate_kernels_real_packed_mt, name="gate_kernels_real_packed") ## 80 gates all packed onto a single vector
			gate_kernels_imag_packed_shared = g.shared_memory_tensor(self.gate_kernels_imag_packed_mt, name="gate_kernels_imag_packed") ## 80 gates all packed onto a single vector

			# concat real and imag parts to pipeline them at once through a shifter
			gate_kernels_packed_mt = g.concat( [gate_kernels_real_packed_shared, gate_kernels_imag_packed_shared], dim=0 )
			gate_kernels_packed_st = gate_kernels_packed_mt.read( streams=g.SG4[0], time=0 )
			print( gate_kernels_packed_st.physical_shape )
			print( gate_kernels_packed_st.shape )

			# split the tensor between hemispheres WEST and EAST (each (2k+1)-th gate kernel goes to the western hemispheres, while 2k-th gates are assigned to the eastern hemisphere
			# the gate kernel elements to be used later are the first 4 elements in each vector
			gate_kernels_imag_list = []
			gate_kernels_real_list = []

			for gate_idx in range(gate_count):

				gate_kernels_packed_list = g.split_inner_splits( gate_kernels_packed_st )
				print(gate_kernels_packed_st.physical_shape)
				gate_kernels_real_packed_st = gate_kernels_packed_list[0]
				gate_kernels_imag_packed_st = gate_kernels_packed_list[1]

				layout_real = layout_gate_kernels_real
				layout_imag = layout_gate_kernels_imag

				gate_kernels_imag_mt = gate_kernels_imag_packed_st.write(name=f"gate_kernel_imag_{gate_idx}", layout=layout_imag + f", A1({gate_count*4 + gate_idx})", program_output=True)
				gate_kernels_real_mt = gate_kernels_real_packed_st.write(name=f"gate_kernel_real_{gate_idx}", layout=layout_real + f", A1({gate_count*4 + gate_idx})", program_output=True)

				# add rule for mutual memory exclusion for the stored real and imaginary parts of the gate kernels
				if ( gate_idx > 0 ):
					g.add_mem_constraints(gate_kernels_imag_list, [gate_kernels_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
					g.add_mem_constraints(gate_kernels_real_list, [gate_kernels_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)    

				gate_kernels_imag_list.append( gate_kernels_imag_mt )
				gate_kernels_real_list.append( gate_kernels_real_mt )
			

				if gate_idx < gate_count-1 :
					permutor_request = permutor_requests[gate_idx % 2]
					shift = 4
					gate_kernels_packed_st = g.shift(gate_kernels_packed_st, 
									shift, 
									permutor_id=permutor_request, 
									shift_src=[inst.NEW_SRC] * 2, 
									dispatch_set=inst.DispatchSet.SET_1, 
									input_streams=[g.SG4[0]], 
									output_streams=[g.SG4[0]])



			# generate gate kernels to be applied on state |0> and resulting to alternating states |0> and |1> according to the periodicity of the current target qubit

			# create distributo maps for the process to broadcast elements 00,01,10,11 of the gate kernel onto a 320 element vectors
			distributor_00_np = np.zeros( (320,), dtype=np.uint8 )
			distribute_map_tensor_mt_00 = g.from_data( np.asarray(distributor_00_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({small_qbit_num_limit})" )
			distributor_01_np = np.ones( (320,), dtype=np.uint8 )
			distribute_map_tensor_mt_01 = g.from_data( np.asarray(distributor_01_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({small_qbit_num_limit+1})" )
			distributor_10_np = np.ones( (320,), dtype=np.uint8 )*2
			distribute_map_tensor_mt_10 = g.from_data( np.asarray(distributor_10_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({small_qbit_num_limit+2})" )
			distributor_11_np = np.ones( (320,), dtype=np.uint8 )*3
			distribute_map_tensor_mt_11 = g.from_data( np.asarray(distributor_11_np, dtype=np.uint8), layout=layout_distribute_map_tensor + f", A1({small_qbit_num_limit+3})" )

			g.add_mem_constraints([distribute_map_tensor_mt_01, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_00], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
			g.add_mem_constraints([distribute_map_tensor_mt_00, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_01], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 
			g.add_mem_constraints([distribute_map_tensor_mt_01, distribute_map_tensor_mt_00, distribute_map_tensor_mt_11], [distribute_map_tensor_mt_10], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 

			distribute_map_tensor_list = [distribute_map_tensor_mt_00, distribute_map_tensor_mt_01, distribute_map_tensor_mt_10, distribute_map_tensor_mt_11]

	  

			gate_kernels_broadcasted_imag_list = []
			gate_kernels_broadcasted_real_list = []
			gate_kernels_broadcasted_imag_copy_list = []
			gate_kernels_broadcasted_real_copy_list = []
			loop_time_long_cycle = 51 # comming from the iterations of the previous cycles (the loops are overlapping in execution)
			loop_time_short_cycle = 31


			for gate_idx in range(gate_count):

				gate_kernel_real_mt = gate_kernels_real_list[gate_idx]
				gate_kernel_imag_mt = gate_kernels_imag_list[gate_idx]

				for idx in range(len(gate_kernel_element_labels) ): # iterates over the labels of the gate kernel elements
					label = gate_kernel_element_labels[idx]

					# concat real and imag parts to pipeline them at once through a broadcaster
					gate_kernel_real_st = gate_kernel_real_mt.read( streams=g.SG4[1], time=12 + 2*idx + ((gate_idx+1)//2)*loop_time_long_cycle + (gate_idx//2)*loop_time_short_cycle ) 
					gate_kernel_imag_st = gate_kernel_imag_mt.read( streams=g.SG4[1], time=14 + 2*idx + ((gate_idx+1)//2)*loop_time_long_cycle + (gate_idx//2)*loop_time_short_cycle )		
					gate_kernel_st = g.concat_inner_splits( [gate_kernel_real_st, gate_kernel_imag_st] )
					distributor_request = distributor_requests[(gate_idx+1) % 2]
					gate_kernel_broadcasted_st = g.distribute_8( gate_kernel_st, 
							distribute_map_tensor_list[idx],
							distributor_req=distributor_request,
							map_stream_req=g.SG1[12],
							bypass8=0b00001111,
					)

					permutor_request = permutor_requests[(gate_idx+1) % 2]
					gate_kernel_broadcasted_st = g.broadcast_lane_0( gate_kernel_broadcasted_st, 
										permutor_request,
										old_bitmap=[inst.NEW_SRC] * 4, 
										mask_bitmap=0b0000,
										input_streams=[g.SG4[3]], 
										output_streams=[g.SG4[2]]
					)

					# split into real ands imaginary parts	
					gate_kernel_broadcasted_list = g.split_inner_splits( gate_kernel_broadcasted_st )
					gate_kernel_broadcasted_real_st = gate_kernel_broadcasted_list[0]
					gate_kernel_broadcasted_imag_st = gate_kernel_broadcasted_list[1]

					######################## save broadcasted elements into memory and add memory exclusion rules #############################


					layout_real = layout_gate_kernels_real
					layout_imag = layout_gate_kernels_imag
					layout_real_copy = layout_gate_kernels_real_copy
					layout_imag_copy = layout_gate_kernels_imag_copy

					address_offset = idx*int(gate_count)

					layout_real = layout_real + f", A1({address_offset + gate_idx})"
					layout_imag = layout_imag + f", A1({address_offset + gate_idx})"
					gate_kernels_broadcasted_real_mt = gate_kernel_broadcasted_real_st.write(name=f"gate_kernel_broadcasted_real_{gate_idx}_"+label, layout=layout_real)#, program_output=True)
					gate_kernels_broadcasted_imag_mt = gate_kernel_broadcasted_imag_st.write(name=f"gate_kernel_broadcasted_imag_{gate_idx}_"+label, layout=layout_imag)#, program_output=True)
					
					# copies of the gate elements to support read concurrency in forthcoming processeses
					layout_real = layout_real_copy + f", A1({address_offset + gate_idx})"
					layout_imag = layout_imag_copy + f", A1({address_offset + gate_idx})"
					gate_kernels_broadcasted_real_copy_mt = gate_kernel_broadcasted_real_st.write(name=f"gate_kernel_broadcasted_real_{gate_idx}_"+label, layout=layout_real)#, program_output=True)
					gate_kernels_broadcasted_imag_copy_mt = gate_kernel_broadcasted_imag_st.write(name=f"gate_kernel_broadcasted_imag_{gate_idx}_"+label, layout=layout_imag)#, program_output=True)

					gate_kernels_broadcasted_real_mt.is_static = True
					gate_kernels_broadcasted_imag_mt.is_static = True
					gate_kernels_broadcasted_real_copy_mt.is_static = True
					gate_kernels_broadcasted_imag_copy_mt.is_static = True				


					# add rule for mutual memory exclusion for the stored real and imaginary parts of the gate kernels  
					if ( len(gate_kernels_broadcasted_real_list) > 0 ):
						g.add_mem_constraints(gate_kernels_broadcasted_real_list, [gate_kernels_broadcasted_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
					if ( len(gate_kernels_broadcasted_imag_list) > 0 ):
						g.add_mem_constraints(gate_kernels_broadcasted_imag_list, [gate_kernels_broadcasted_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)   
					if ( len(gate_kernels_broadcasted_real_copy_list) > 0 ):
						g.add_mem_constraints(gate_kernels_broadcasted_real_copy_list, [gate_kernels_broadcasted_real_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)     
					if ( len(gate_kernels_broadcasted_imag_copy_list) > 0 ):
						g.add_mem_constraints(gate_kernels_broadcasted_imag_copy_list, [gate_kernels_broadcasted_imag_copy_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)   					
		
		
					g.add_mem_constraints(gate_kernels_real_list, [gate_kernels_broadcasted_real_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE)    
					g.add_mem_constraints(gate_kernels_imag_list, [gate_kernels_broadcasted_imag_mt], g.MemConstraintType.NOT_MUTUALLY_EXCLUSIVE) 


					gate_kernels_broadcasted_real_list.append( gate_kernels_broadcasted_real_mt )
					gate_kernels_broadcasted_imag_list.append( gate_kernels_broadcasted_imag_mt )
					self.gate_kernels_broadcasted_real_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_real_mt
					self.gate_kernels_broadcasted_imag_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_imag_mt
					print( f"{gate_idx}_"+label )


					gate_kernels_broadcasted_real_copy_list.append( gate_kernels_broadcasted_real_copy_mt )
					gate_kernels_broadcasted_imag_copy_list.append( gate_kernels_broadcasted_imag_copy_mt )
					self.gate_kernels_broadcasted_real_copy_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_real_copy_mt
					self.gate_kernels_broadcasted_imag_copy_dict[f"{gate_idx}_"+label] = gate_kernels_broadcasted_imag_copy_mt


			self.pgm_pkg.compile_program_context(pgm2)

			g.write_visualizer_data("QCsim_"+name)	
