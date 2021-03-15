#macro(cuda_compile_and_embed output_var cuda_file)
#  set(c_var_name ${output_var})
#  cuda_compile_ptx(ptx_files ${cuda_file})
#  list(GET ptx_files 0 ptx_file)
#  set(embedded_file ${ptx_file}_embedded.c)
#  add_custom_command(
#	  OUTPUT ${embedded_file}
#	  COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
#	  DEPENDS ${ptx_file}
#	  COMMENT "compiling (and embedding ptx from) ${cuda_file}"
#  )
#  set(${output_var} ${embedded_file})
#endmacro()