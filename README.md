# Distributed memory
The aim of this project is to use MPI for parallelism on a distributed memory 
architecture to implement the relaxation technique for solving differential 
equations. This is achieved by having an array of values and repeatedly 
replacing values with the average of its four neighbours, except boundary 
values which remain fixed, until all values settle down to a given precision. 
The solution is written in C using MPI.

## Compile
Compile the program using:  
`mpicc -Wall -Wextra -Wconversion -std=gnu99 -o prog prog.c`  

## Run
Run the program using:  
`mpirun -np 2 ./prog -d10 -p0.1`  

Use flags  
	`-d` to specify the dimensions of the array  
	`-p` to specify the required precision  
	
## Debug

To print array values after each iteration,  
Compile the program in debug mode using:  
`mpicc -Wall -Wextra -Wconversion prog.c -std=gnu99 -o prog -DDEBUG`

Then run the program normally.

## Correctness

To print only the final relaxed array,  
Compile the program in correctness mode using  
`mpicc -Wall -Wextra -Wconversion prog.c -std=gnu99 -o prog -DCORRECTNESS`

Then run the program normally.

### Check output

After running the program in correctness mode, run  
`check_output.sh p d` 

where  
`p` is the path to the results directory  
`d` is the number of dimensions of the array  
