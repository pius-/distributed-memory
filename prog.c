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
 * Populates the arrays 'a' and 'b'
 * with 1s for border cells, and 0s for inner cells.
 * Initially both 'a' and 'b' will be the same.
 */
void populate_array(double **a, double **b, int dimensions)
{
	for (int i = 0; i < dimensions; i++)
	{
		a[i][0] = 1.0;				// left border
		a[0][i] = 1.0;				// top border
		a[i][dimensions - 1] = 1.0;	// right border
		a[dimensions - 1][i] = 1.0;	// bottom border

		b[i][0] = 1.0;
		b[0][i] = 1.0;
		b[i][dimensions - 1] = 1.0;
		b[dimensions - 1][i] = 1.0;
	}
}

char relax_section(double **a, double **b, int dimensions, double precision,
		int cells_to_relax)
{
	char is_done = 1;

	// first and last rows don't need relaxing
	for (int i = 1; i < dimensions - 1; i++)
	{
		for (int j = 0; j < dimensions && cells_to_relax; j++)
		{
			cells_to_relax--;

			// first and last columns don't need relaxing
			// but have to be included in results for gather v
			if (j == 0 || j == dimensions - 1)
			{
				b[i][j] = a[i][j];
			}
			else
			{
				b[i][j] = (a[i - 1][j]
						+ a[i + 1][j]
						+ a[i][j - 1]
						+ a[i][j + 1]) / 4;

				// precision values are only calculated until a cell is found
				// with a precision value greater than the required precision
				// after which we know the section is not relaxed
				// so there is no point checking other cell precisions
				if (is_done && fabs(b[i][j] - a[i][j]) > precision)
				{
					is_done = 0;
				}
			}
		}
	}

	return is_done;
}

void relax_array(double **a, double **b,
		int rank, int dimensions, double precision,
		int *send_counts, int *send_displs,
		int *recv_counts, int *recv_displs)
{
	int iterations = 0;
	char global_done = 0, local_done = 0, root = 0;
	while (!global_done)
	{
		iterations++;

		MPI_Scatterv(&(a[0][0]), send_counts, send_displs,
				MPI_DOUBLE, &(a[0][0]), send_counts[rank],
				MPI_DOUBLE, root, MPI_COMM_WORLD);

		// each processor relaxes its section, and stores results in 'b'
		local_done = relax_section(a, b, dimensions, precision,
				recv_counts[rank]);

		// gather results from each processor's 'b' array
		// into the root processors 'a' array
		// row 0 is used for calculations, results start from row 1
		MPI_Gatherv(&(b[1][0]), recv_counts[rank], MPI_DOUBLE,
				&(a[0][0]), recv_counts, recv_displs, MPI_DOUBLE, root,
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

	if (rank == root)
	{
		printf("iterations: %d\n", iterations);
	}
}

void alloc_work(int dimensions, int processors,
		int *send_counts, int *send_displs, int *recv_counts, int *recv_displs)
{
	// each processor will relax n rows
	int nrows = (dimensions - 2) / processors;

	// first m processors will relax nrows + 1 rows, where m is extra_rows
	int extra_rows = (dimensions - 2) % processors;

	// first processor will start at row 1, after n (= dimensions) cells
	int displacement = dimensions;

	for (int rank = 0; rank < processors; rank++)
	{
		// first m rows will relax n + 1 rows
		// multiply by dimensions to get total number of cells
		recv_counts[rank] = (nrows + (rank < extra_rows)) * dimensions;

		// number of cells to send, includes row above and row below
		send_counts[rank] = recv_counts[rank] + 2 * dimensions;

		// calculate start position, displacement relative to start
		recv_displs[rank] = displacement;
		displacement += recv_counts[rank];

		send_displs[rank] = recv_displs[rank] - dimensions;
	}
}

void alloc_memory(int dimensions, int processors, int rank,
		double ***a, double ***b, double **a_buf, double **b_buf,
		int **send_counts, int **send_displs,
		int **recv_counts, int **recv_displs)
{
	*a = malloc((unsigned long)dimensions * sizeof(double *));
	*b = malloc((unsigned long)dimensions * sizeof(double *));

	int rows = dimensions, root = 0;
	if (rank != root)
	{
		// each processor will relax n rows,
		// first m processors will relax n + 1 rows,
		// + 2 extra for row above and below for relaxation calculations
		int nrows = dimensions / processors;
		int extra_rows = (dimensions - 2) % processors;
		rows = nrows + (rank < extra_rows) + 2;
	}

	*a_buf = calloc((unsigned long)(dimensions * rows), sizeof(double));
	*b_buf = calloc((unsigned long)(dimensions * rows), sizeof(double));

	*send_counts = malloc((unsigned long)processors * sizeof(int));
	*send_displs = malloc((unsigned long)processors * sizeof(int));
	*recv_counts = malloc((unsigned long)processors * sizeof(int));
	*recv_displs = malloc((unsigned long)processors * sizeof(int));

	if (*a == NULL || *b == NULL || *a_buf == NULL || *b_buf == NULL
			|| send_counts == NULL || send_displs == NULL
			|| recv_counts == NULL || recv_displs == NULL)
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
		int *send_counts, int *send_displs,
		int *recv_counts, int *recv_displs)
{
	free(a);
	free(b);
	free(a_buf);
	free(b_buf);
	free(send_counts);
	free(send_displs);
	free(recv_counts);
	free(recv_displs);
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
		printf("error starting MPI program.\n");
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
	int *send_counts = NULL, *send_displs = NULL,
		*recv_counts = NULL, *recv_displs = NULL;

	alloc_memory(dimensions, processors, rank, &a, &b, &a_buf, &b_buf,
			&send_counts, &send_displs, &recv_counts, &recv_displs);

	alloc_work(dimensions, processors,
			send_counts, send_displs, recv_counts, recv_displs);

	if (rank == root)
	{
		populate_array(a, b, dimensions);
#ifdef DEBUG
		print_array(a, dimensions);
#endif
	}

	relax_array(a, b, rank, dimensions, precision,
			send_counts, send_displs, recv_counts, recv_displs);

	dealloc_memory(a, b, a_buf, b_buf,
			send_counts, send_displs, recv_counts, recv_displs);

	MPI_Finalize();
	exit(EXIT_SUCCESS);
}
