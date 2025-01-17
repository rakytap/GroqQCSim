# GroqQCSim

- Instructions for Groq-SQUANDER integration:

git clone GroqQCSim (branch gm_u3_gte8) to $
cd GroqQCSim
python state_transform.py
# Ensure IOP name is QCsim_multi_program.<num_qubits>.<max_gates>.iop e.g.
mv build_iop/QCsim_multi_program.0.iop build_iop/QCsim_multi_program.17.768.iop
make all  # can test with make test & build/testsv
cd ..

git clone SQUANDER (branch gm_dfe_sv) to $
# Build SQUANDER e.g. on the Groq server:
export TBB_LIB_DIR=/usr/lib/x86_64-linux-gnu 
export TBB_INC_DIR=/usr/include
export QGD_DFE=1
python setup.py build_ext

# Adjust examples/VQE/Heisenberg_VQE.py for qbit_num = 17
LD_LIBRARY_PATH=$/GroqQCSim/build:$LD_LIBRARY_PATH python examples/VQE/Heisenberg_VQE.py