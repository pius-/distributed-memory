#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

void print_array(double **a, int dimensions)
{
	for (int i = 0; i < dimensions; i++)
	{
		for (int j = 0; j < dimensions; j++)
		{
			printf("%f\t", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

/*
 * Populates the array with values into 'a' and 'b'.
 * Initially both 'a' and 'b' will be the same.
 */
void populate_array(double **a, double **b, int dimensions)
{
	for (int i = 0; i < dimensions; i++)
	{
		for (int j = 0; j < dimensions; j++)
		{
			// using rand generates the same random values on multiple runs
			// as it uses the same seed
			int val = rand() % 10;
			a[i][j] = val;
			b[i][j] = val;
		}
	}
}

char relax_section(double **a, double **b, int dimensions, double precision,
				   int start_row, int rows_to_relax)
{
	char is_done = 1;

	for (int i = start_row;
		 i < dimensions - 1 && i < start_row + rows_to_relax;
		 i++)
	{
		for (int j = 1; j < dimensions - 1; j++)
		{
			b[i][j] = (a[i - 1][j] 
				+ a[i + 1][j] 
				+ a[i][j - 1] 
				+ a[i][j + 1]) / 4;

			if (is_done && fabs(b[i][j] - a[i][j]) > precision)
			{
				is_done = 0;
			}
		}
	}

	return is_done;
}

void relax_array(double **a, double **b, 
				 int rank, int dimensions, double precision,
				 int *start_rows, int *rows_to_relax, 
				 int *recvcounts, int *displs)
{
	char global_done = 0, local_done = 0, root = 0;
	while (!global_done)
	{
		// broadcast so all processors have the same starting array
		MPI_Bcast(&a[0][0], dimensions * dimensions,
				  MPI_DOUBLE, root, MPI_COMM_WORLD);

		// each processor relaxes its section, and stores results in 'b'
		local_done = relax_section(a, b, dimensions, precision,
								   start_rows[rank], rows_to_relax[rank]);

		// gather results from each processor's 'b' array
		// into the root processors 'a' array
		MPI_Gatherv(&b[start_rows[rank]][1], recvcounts[rank], MPI_DOUBLE,
					&a[0][0], recvcounts, displs, MPI_DOUBLE, root, 
					MPI_COMM_WORLD);

		// bitwise and of local_done to find global_done
		// local_done will be 0 if a processor is not done
		// global_done will only be 1, if all processors are done
		MPI_Allreduce(&local_done, &global_done, 1,
					  MPI_CHAR, MPI_BAND, MPI_COMM_WORLD);

#ifdef DEBUG
		if (rank == root)
		{
			print_array(a, dimensions);
		}
#endif

	}
}

void alloc_work(int dimensions, int processors,
				int *start_rows, int *rows_to_relax, 
				int *recvcounts, int *displs)
{
	// each processor will relax n rows
	int nrows = (dimensions - 2) / processors;

	// first m processors will relax n + 1 rows,
	// where m is extra_rows and n is rows_to_relax
	int extra_rows = nrows % processors;

	// first processor will start at row 1
	int row = 1;

	for (int rank = 0; rank < processors; rank++)
	{
		// first m rows will relax n + 1 rows
		rows_to_relax[rank] = nrows + (rank < extra_rows);

		// calculate start position for next processor
		start_rows[rank] = row;
		row += rows_to_relax[rank];

		// number of elements to relax
		recvcounts[rank] = rows_to_relax[rank] * dimensions;

		// displacement relative to start
		displs[rank] = start_rows[rank] * dimensions + 1;
	}
}

void alloc_memory(int dimensions, int processors,
				  double ***a, double ***b, double **a_buf, double **b_buf,
				  int **start_rows, int **rows_to_relax, 
				  int **recvcounts, int **displs)
{
	*a = malloc((unsigned long)dimensions * sizeof(double *));
	*b = malloc((unsigned long)dimensions * sizeof(double *));

	*a_buf = malloc(
		(unsigned long)(dimensions * dimensions) * sizeof(double));
	*b_buf = malloc(
		(unsigned long)(dimensions * dimensions) * sizeof(double));

	*start_rows = malloc((unsigned long)processors * sizeof(int));
	*rows_to_relax = malloc((unsigned long)processors * sizeof(int));
	*recvcounts = malloc((unsigned long)processors * sizeof(int));
	*displs = malloc((unsigned long)processors * sizeof(int));

	if (*a == NULL || *b == NULL || *a_buf == NULL || *b_buf == NULL 
		|| start_rows == NULL || rows_to_relax == NULL 
		|| recvcounts == NULL || displs == NULL)
	{
		printf("malloc failed.\n");
		exit(EXIT_FAILURE);
	}

	// each a[i] points to start of a row
	for (int i = 0; i < dimensions; i++)
	{
		(*a)[i] = *a_buf + dimensions * i;
		(*b)[i] = *b_buf + dimensions * i;
	}
}

void dealloc_memory(double **a, double **b, double *a_buf, double *b_buf,
		int *start_rows, int *rows_to_relax, 
		int *recvcounts, int *displs)
{
	free(a);
	free(b);
	free(a_buf);
	free(b_buf);
	free(start_rows);
	free(rows_to_relax);
	free(recvcounts);
	free(displs);
}

void process_args(int argc, char *argv[], int *dimensions, double *precision)
{
	if (argc != 3)
	{
		printf("unexpected number of arguments.\n");
		exit(EXIT_FAILURE);
	}

	int opt;
	const char *optstring = "d:p:";
	while ((opt = getopt(argc, argv, optstring)) != -1)
	{
		switch (opt)
		{
		case 'd':
			*dimensions = atoi(optarg);
			break;
		case 'p':
			*precision = atof(optarg);
			break;
		default:
			printf("unexpected argument.\n");
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char *argv[])
{
	int rc, rank, root = 0, processors; //, namelen;

	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS)
	{
		printf("Error starting MPI program\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processors);

	if (rank == root)
	{
		printf("main reports %d procs\n", processors);
	}

	int dimensions = 0;
	double precision = 1;
	process_args(argc, argv, &dimensions, &precision);

	if (rank == root)
	{
		printf("using dimension: %d\n", dimensions);
		printf("using processors: %d\n", processors);
		printf("using precision: %lf\n", precision);
	}

	double **a = NULL, **b = NULL, *a_buf = NULL, *b_buf = NULL;
	int *start_rows = NULL, *rows_to_relax = NULL,
		*recvcounts = NULL, *displs = NULL;

	alloc_memory(dimensions, processors,
				 &a, &b, &a_buf, &b_buf,
				 &start_rows, &rows_to_relax, &recvcounts, &displs);

	alloc_work(dimensions, processors,
			   start_rows, rows_to_relax, recvcounts, displs);

	if (rank == root)
	{
		populate_array(a, b, dimensions);
#ifdef DEBUG
		print_array(a, dimensions);
#endif
	}

	relax_array(a, b, rank, dimensions, precision,
				start_rows, rows_to_relax, recvcounts, displs);

	dealloc_memory(a, b, a_buf, b_buf, 
			start_rows, rows_to_relax, recvcounts, displs);

	MPI_Finalize();
	exit(EXIT_SUCCESS);
}
