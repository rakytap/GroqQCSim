from split_gates import Split_gates
from memory_layouts import *
import groq.api as g
import numpy as np


	

class Gate_counters(Split_gates):
	"""
	A base class used to implement the gate operation on qubits S.
	S qubits are qubits 0,1,2,...,7


	Attributes
	----------
    
	mxm_permutation : bool
		If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
		Permutor takes twice as much time as the MXM.

	"""

	def __init__(self, pgm_pkg, mxm_permutation=True):
		"""
		Parameters
		----------
		pgm_pkg : ProgramPackage
			The program package in which the gate operation program is generated
		mxm_permutation : bool
			If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
			Permutor takes twice as much time as the MXM.
		"""
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
		
		
	

	def increment_gate_counters( self ):
				
		#### increase the gate counter for permute map selector ####
		alu_rq_12 = g.tensor.create_alu_request( alus=[12] )
				
		constant_1_address_offset_st     = self.constant_1_address_offset_mt.read(streams=[g.SG1_E[0]])				

		gate_counter_perm_map_selector_shared = g.shared_memory_tensor(self.gate_counter_perm_map_selector_mt, name="gate_counter_perm_map_selector_shared" )
		gate_counter_perm_map_selector_st  = gate_counter_perm_map_selector_shared.read(streams=[g.SG1_E[4]])
				
		# offset the memory address by 1
		gate_counter_perm_map_selector_st = g.add(constant_1_address_offset_st, gate_counter_perm_map_selector_st, alus=alu_rq_12, output_streams=g.SG4_W[1])
				
		# save the offsetted memory address
		gate_counter_perm_map_selector_mt = gate_counter_perm_map_selector_st.write(name=f"gate_counter_perm_map_selector_mt", \
													 storage_req=gate_counter_perm_map_selector_shared.storage_request, 
													 time=0)#, program_output=True)
				

		#### increase the gate counter for gate filtering selector ####
		alu_rq_15 = g.tensor.create_alu_request( alus=[15] )			
			
				
		#constant_1_address_offset_st     = self.constant_1_address_offset_mt.read(streams=[g.SG1_E[0]])				

		gate_counter_gate_filtering_selector_shared = g.shared_memory_tensor(self.gate_counter_gate_filtering_selector_mt,   \
												   	name="gate_counter_gate_filtering_selector_shared") 
		gate_counter_gate_filtering_selector_st  = gate_counter_gate_filtering_selector_shared.read(streams=[g.SG1_W[4]])
				
		# offset the memory address by 1
		gate_counter_gate_filtering_selector_st = g.add(constant_1_address_offset_st, gate_counter_gate_filtering_selector_st, alus=alu_rq_15, output_streams=g.SG4_E[6])
				
		# save the offsetted memory address
		gate_counter_gate_filtering_selector_mt = gate_counter_gate_filtering_selector_st.write( \
								name=f"gate_counter_gate_filtering_selector_mt", \
								storage_req=gate_counter_gate_filtering_selector_shared.storage_request, 
								time=None)#, program_output=True)




				
		#### increase the gate counter for gate element selectors ####
		alu_rq_0  = g.tensor.create_alu_request( alus=[0] )	
		alu_rq_1  = g.tensor.create_alu_request( alus=[1] )	
		alu_rq_13 = g.tensor.create_alu_request( alus=[13] )	
		alu_rq_14 = g.tensor.create_alu_request( alus=[14] )	

		constant_1_address_offset_st_copy     = self.constant_1_address_offset_mt.read(streams=[g.SG1_E[0]])				

		gate_counter_gate_element_selector_00_shared = g.shared_memory_tensor( \
							self.gate_counter_gate_element_selector_00_mt, \
							name="gate_counter_gate_element_selector_00_shared") 
				
		gate_counter_gate_element_selector_01_shared = g.shared_memory_tensor( \
							self.gate_counter_gate_element_selector_01_mt, \
							name="gate_counter_gate_element_selector_01_shared") 

				
		gate_counter_gate_element_selector_10_shared = g.shared_memory_tensor( \
							self.gate_counter_gate_element_selector_10_mt, \
							name="gate_counter_gate_element_selector_10_shared") 
				
		gate_counter_gate_element_selector_11_shared = g.shared_memory_tensor( \
							self.gate_counter_gate_element_selector_11_mt, \
							name="gate_counter_gate_element_selector_11_shared") 
				
		gate_counter_gate_element_selector_00_st  = gate_counter_gate_element_selector_00_shared.read(streams=[g.SG1_W[8]])
		gate_counter_gate_element_selector_11_st  = gate_counter_gate_element_selector_11_shared.read(streams=[g.SG1_W[12]])
		gate_counter_gate_element_selector_01_st  = gate_counter_gate_element_selector_01_shared.read(streams=[g.SG1_W[24]])
		gate_counter_gate_element_selector_10_st  = gate_counter_gate_element_selector_10_shared.read(streams=[g.SG1_W[28]])

		#gate_counter_gate_element_selector_11_st.write( name='wwww', layout='H1(W), -1, S1(42)')#, program_output=True )
				
		# offset the memory address by 1
		gate_counter_gate_element_selector_00_st = g.add(constant_1_address_offset_st, gate_counter_gate_element_selector_00_st, alus=alu_rq_0, output_streams=g.SG4_E[3])
		gate_counter_gate_element_selector_11_st = g.add(constant_1_address_offset_st, gate_counter_gate_element_selector_11_st, alus=alu_rq_1, output_streams=g.SG4_E[2])
		gate_counter_gate_element_selector_01_st = g.add(constant_1_address_offset_st_copy, gate_counter_gate_element_selector_01_st, alus=alu_rq_14, output_streams=g.SG4_E[7])
		gate_counter_gate_element_selector_10_st = g.add(constant_1_address_offset_st_copy, gate_counter_gate_element_selector_10_st, alus=alu_rq_13, output_streams=g.SG4_E[0])
				
		#gate_counter_gate_element_selector_11_st.write( name='rrrr', layout='H1(E), -1, S1(42)')#, program_output=True )


		# save the offsetted memory address
		gate_counter_gate_element_selector_00_mt = gate_counter_gate_element_selector_00_st.write( \
								name=f"gate_counter_gate_element_selector_00_mt", \
								storage_req=gate_counter_gate_element_selector_00_shared.storage_request, 
								time=None)#, program_output=False)

		gate_counter_gate_element_selector_11_mt = gate_counter_gate_element_selector_11_st.write( \
								name=f"gate_counter_gate_element_selector_11_mt", \
								storage_req=gate_counter_gate_element_selector_11_shared.storage_request, 
								time=None)#, program_output=True)


		gate_counter_gate_element_selector_01_mt = gate_counter_gate_element_selector_01_st.write( \
								name=f"gate_counter_gate_element_selector_01_mt", \
								storage_req=gate_counter_gate_element_selector_01_shared.storage_request, 
								time=2)#, program_output=True)


		gate_counter_gate_element_selector_10_mt = gate_counter_gate_element_selector_10_st.write( \
								name=f"gate_counter_gate_element_selector_10_mt", \
								storage_req=gate_counter_gate_element_selector_10_shared.storage_request, 
								time=None)#, program_output=True) 
					
				

		#### increase the gate counter for S control qubit selector ####

		alu_rq_7 = g.tensor.create_alu_request( alus=[7] )
				
		constant_1_address_offset_st     = self.constant_1_address_offset_mt.read(streams=[g.SG1_E[8]])	

		gate_counter_S_control_qbit_selector_shared = g.shared_memory_tensor(self.gate_counter_S_control_qbit_selector_mt,   \
   										name="gate_counter_S_control_qbit_selector_shared") 
		gate_counter_S_control_qbit_selector_st     = gate_counter_S_control_qbit_selector_shared.read( streams=g.SG1[12] )

							
		# offset the memory address by 1
		gate_counter_S_control_qbit_selector_st = g.add(constant_1_address_offset_st, gate_counter_S_control_qbit_selector_st, alus=alu_rq_7, output_streams=g.SG4_E[4])
				
		# save the offsetted memory address
		gate_counter_S_control_qbit_selector_mt = gate_counter_S_control_qbit_selector_st.write(name=f"gate_counter_S_control_qbit_selector_mt", \
											 storage_req=gate_counter_S_control_qbit_selector_shared.storage_request, 
											 time=2)#, program_output=True) 



		#### increase the gate counter for L/H control qubit distribute map selector ####


		alu_rq_6 = g.tensor.create_alu_request( alus=[6] )
				
		#constant_1_address_offset_st     = self.constant_1_address_offset_mt.read(streams=[g.SG1_E[8]])	

		gate_counter_distributor_maps_west_shared = g.shared_memory_tensor(self.gate_counter_distributor_maps_west_mt,   \
   										name="gate_counter_S_control_qbit_selector_shared") 
		gate_counter_distributor_maps_west_st     = gate_counter_distributor_maps_west_shared.read( streams=g.SG1[16] )

							
		# offset the memory address by 1
		gate_counter_distributor_maps_west_st = g.add(constant_1_address_offset_st, gate_counter_distributor_maps_west_st, alus=alu_rq_6, output_streams=g.SG4_W[4])
				
		# save the offsetted memory address
		gate_counter_distributor_maps_west_mt = gate_counter_distributor_maps_west_st.write(name=f"gate_counter_distributor_maps_west_mt", \
											 storage_req=gate_counter_distributor_maps_west_shared.storage_request, 
											 time=None)#, program_output=True) 
											 
											 
											 
											 
											 
											 
									 
		#### increase the gate counter for L/H target qubit distribute map selector ####
		
		
		alu_rq_5 = g.tensor.create_alu_request( alus=[5] )
				
		constant_1_address_offset_st     = self.constant_1_address_offset_mt.read(streams=[g.SG1_E[8]])	

		gate_counter_distributor_maps_st     = self.gate_counter_target_qubit_L_distributor_maps_write_mt.read( streams=g.SG1[16] )

							
		# offset the memory address by 1
		gate_counter_distributor_maps_st = g.add(constant_1_address_offset_st, gate_counter_distributor_maps_st, alus=alu_rq_5, output_streams=g.SG4_W[4])
				
		# save the offsetted memory address
		gate_counter_distributor_maps_west_mt = gate_counter_distributor_maps_st.write(name=f"gate_counter_distributor_maps_west_mt", \
											 storage_req=self.gate_counter_target_qubit_L_distributor_maps_write_mt.storage_request, 
											 time=3)#, program_output=True) 											 
											 
										 
											 
											


