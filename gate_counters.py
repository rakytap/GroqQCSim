from generate_static_data import Generate_static_data
from memory_layouts import *
import groq.api as g
import numpy as np


	

class Gate_counters(Generate_static_data):
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

		# prepare the (masked) increment for the higher bits of the gate counters
		alu_rq_9 = g.tensor.create_alu_request( alus=[9] )
		alu_rq_8 = g.tensor.create_alu_request( alus=[8] )
		alu_rq_5 = g.tensor.create_alu_request( alus=[5] )
		gate_counter_address_offset_higher_bits_shared = g.shared_memory_tensor(self.gate_counter_address_offset_higher_bits_mt, 
                                                                               name="gate_counter_address_offset_higher_bits_shared" )


		constant_1_address_offset_higher_bits_st   = self.constant_1_address_offset_higher_bits_mt.read(streams=[g.SG1_E[16]])
		gate_counter_address_offset_higher_bits_st = gate_counter_address_offset_higher_bits_shared.read(streams=[g.SG1_E[20]])	

		gate_counter_address_offset_higher_bits_st = g.add(constant_1_address_offset_higher_bits_st, 
                                                          gate_counter_address_offset_higher_bits_st, 
                                                          alus=alu_rq_8, 
                                                          output_streams=g.SG4_E[5], 
                                                          overflow_mode=g.OverflowMode.MODULAR)
		
		constant_1_address_offset_higher_bits_st   = self.constant_1_address_offset_higher_bits_mt.read(streams=[g.SG1_E[16]])

		masked_1_address_offset_higher_bits_st     = g.mask_bar( gate_counter_address_offset_higher_bits_st,
                                                                         constant_1_address_offset_higher_bits_st,
                                                                         alus=alu_rq_9,
                                                                         output_streams=g.SG4_W[5])


		masked_1_address_offset_higher_bits_mt = masked_1_address_offset_higher_bits_st.write(
                                                             name="masked_address_offset", 
                                                             layout="H1(W), -1, S1(22)",
                                                             time = 0)#, program_output=True  )
                                        
				

		gate_counter_address_offset_higher_bits_st = g.vxm_identity(gate_counter_address_offset_higher_bits_st, 
                                                                            alus=alu_rq_5, 
                                                                            output_streams=g.SG4_W[4])


		gate_counter_address_offset_higher_bits_st.write(name=f"gate_counter_address_offset_higher_bits", 
						storage_req=gate_counter_address_offset_higher_bits_shared.storage_request)


		# 3 west side gate counters
		gate_counter_perm_map_selector_shared = g.shared_memory_tensor(self.gate_counter_perm_map_selector_mt, 
                                                                               name="gate_counter_perm_map_selector_shared" )

		


		gate_counter_distributor_maps_west_shared = g.shared_memory_tensor(
                                                                self.gate_counter_distributor_maps_west_mt,   \
   								name="gate_counter_S_control_qbit_selector_shared") 


		gate_counter_target_qubit_L_distributor_maps_write_shared = g.shared_memory_tensor(
                                                                              self.gate_counter_target_qubit_L_distributor_maps_write_mt,
                                                                              name="gate_counter_target_qubit_L_distributor_maps_write_shared" )

		gate_counters_west_mt = g.stack( [gate_counter_perm_map_selector_shared,
                                                  gate_counter_distributor_maps_west_shared,
                                                  gate_counter_target_qubit_L_distributor_maps_write_shared], 
                                                  dim=0 )

		alu_rq_12 = g.tensor.create_alu_request( alus=[12] ) # for low bits
		alu_rq_13 = g.tensor.create_alu_request( alus=[13] ) # for high bits
				
		constant_1_address_offset_st           = self.constant_1_address_offset_lower_bits_mt.read(streams=[g.SG1_E[0]])	
		masked_1_address_offset_higher_bits_st = masked_1_address_offset_higher_bits_mt.read(streams=[g.SG1_E[24]])	
		gate_counters_west_st                  = gate_counters_west_mt.read(streams=[g.SG1_E[4]])
				
		# offset the memory address by 1
		gate_counters_west_st        = g.add(constant_1_address_offset_st, 
                                                          gate_counters_west_st, 
                                                          alus=alu_rq_12, 
                                                          output_streams=g.SG4_E[1], 
                                                          overflow_mode=g.OverflowMode.MODULAR)

		# offset the memory address by 0 or 1  on high bits according to the modular addition on the low bits
		gate_counters_west_st        = g.add(masked_1_address_offset_higher_bits_st, 
                                                          gate_counters_west_st, 
                                                          alus=alu_rq_13, 
                                                          output_streams=g.SG4_W[1], 
                                                          overflow_mode=g.OverflowMode.MODULAR)
				
		# save the offsetted memory address
		gate_counters_west_st.write(name=f"gate_counters_west", \
						storage_req=gate_counters_west_mt.storage_request, 
						time=28)#, program_output=True)



		
		# 6 east side gate counters

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


		gate_counter_S_control_qbit_selector_shared = g.shared_memory_tensor(
                                                                 self.gate_counter_S_control_qbit_selector_mt,   \
   								 name="gate_counter_S_control_qbit_selector_shared") 


		gate_counter_gate_filtering_selector_shared = g.shared_memory_tensor(
                                                                  self.gate_counter_gate_filtering_selector_mt,   \
                                                                  name="gate_counter_gate_filtering_selector_shared") 


		gate_counters_east_mt = g.stack( [gate_counter_gate_filtering_selector_shared,
                                                  gate_counter_S_control_qbit_selector_shared,
                                                  gate_counter_gate_element_selector_00_shared,
                                                  gate_counter_gate_element_selector_01_shared,
                                                  gate_counter_gate_element_selector_10_shared,
                                                  gate_counter_gate_element_selector_11_shared],       
                                                  dim=0 )




		alu_rq_15 = g.tensor.create_alu_request( alus=[15] ) # for low bits		
		alu_rq_14 = g.tensor.create_alu_request( alus=[14] ) # for high bits
			
				
		constant_1_address_offset_st           = self.constant_1_address_offset_lower_bits_mt.read(streams=[g.SG1_E[0]])	
		masked_1_address_offset_higher_bits_st = masked_1_address_offset_higher_bits_mt.read(streams=[g.SG1_E[28]])				

		gate_counters_east_st  = gate_counters_east_mt.read(streams=[g.SG1_W[4]])
				
		# offset the memory address by 1
		gate_counters_east_st  = g.add(constant_1_address_offset_st, 
                                                          gate_counters_east_st, 
                                                          alus=alu_rq_15, 
                                                          output_streams=g.SG4_W[6],
                                                          overflow_mode=g.OverflowMode.MODULAR)

		gate_counters_east_st  = g.add(masked_1_address_offset_higher_bits_st, 
                                                          gate_counters_east_st, 
                                                          alus=alu_rq_14, 
                                                          output_streams=g.SG4_E[6],
                                                          overflow_mode=g.OverflowMode.MODULAR)
				
		# save the offsetted memory address
		gate_counters_east_st.write( name=f"gate_counters_east", 
					     storage_req=gate_counters_east_mt.storage_request, 
					     time=27)#, program_output=True)
		
			

		

