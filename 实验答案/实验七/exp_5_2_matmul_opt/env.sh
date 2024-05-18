if [ -z "${NEUWARE_HOME}" ]; then
  export NEUWARE_HOME=/torch/neuware_home
fi

export PATH=${NEUWARE_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib/llvm-mm-cxx11-old-abi/lib:${LD_LIBRARY_PATH}

