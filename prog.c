#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>

void handle_rc(int rc, char *error_message)
{
	if (rc != MPI_SUCCESS)
	{
		printf("%s\n", error_message);
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
}

void swap_array(double ***a, double ***b)
{
	double **temp = *a;
	*a = *b;
	*b = temp;
}

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
		a[i][dimensions - 1] = 1.0; // right border
		a[dimensions - 1][i] = 1.0; // bottom border

		b[i][0] = 1.0;
		b[0][i] = 1.0;
		b[i][dimensions - 1] = 1.0;
		b[dimensions - 1][i] = 1.0;
	}
}

char relax_section(double **a, double **b, int rows, int columns,
		double precision)
{
	char is_done = 1;

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < columns; j++)
		{
			// first and last rows and columns don't need relaxing
			// but have to be included in results
			// because the arrays will be swapped for the next iteration
			if (i == 0 || i == rows - 1 || j == 0 || j == columns - 1)
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

void relax_array(double **a, double **b, int my_rank, int root,
		int dimensions, int processors, double precision,
		int my_send_count, int my_recv_count,
		int *send_counts, int *send_displs,
		int *recv_counts, int *recv_displs)
{
	int rc, iterations = 0;

	// rows and columns in this processors section
	int rows = my_send_count / dimensions;
	int columns = dimensions;

	// index of penultimate row and last row in section
	int pen_row = rows - 2;
	int last_row = rows - 1;

	// rank of processors to the left and to the right
	int proc_left_rank = my_rank - 1;
	int proc_right_rank = my_rank + 1;

	MPI_Request send_left_req, send_right_req, recv_left_req, recv_right_req;

	rc = MPI_Scatterv(&(a[0][0]), send_counts, send_displs,
			MPI_DOUBLE, &(a[0][0]), my_send_count,
			MPI_DOUBLE, root, MPI_COMM_WORLD);
	handle_rc(rc, "error scattering data.");

	char global_done = 0, local_done = 0;
	while (!global_done)
	{
		if (my_rank == root)
		{
			iterations++;
		}

		// each processor relaxes its section, and stores results in 'b'
		local_done = relax_section(a, b, rows, columns, precision);

		// async send and receive, will not deadlock
		// data to be sent doesnt overlap data to be received
		// MPI_reduce at the end will synchronise all processors
		// MPI_wait at the end will ensure data is ready for next iteration

		// first proc doesnt need to send/recv left
		if (my_rank != 0)
		{
			rc = MPI_Isend(&(b[1][0]), dimensions, MPI_DOUBLE,
					proc_left_rank, 1, MPI_COMM_WORLD, &send_left_req);
			handle_rc(rc, "error sending data to the left.");

			rc = MPI_Irecv(&(b[0][0]), dimensions, MPI_DOUBLE,
					proc_left_rank, 2, MPI_COMM_WORLD, &recv_left_req);
			handle_rc(rc, "error receiving data from the left.");
		}

		// last processor doesnt need to send/recv right
		if (my_rank != processors - 1)
		{
			rc = MPI_Isend(&(b[pen_row][0]), dimensions, MPI_DOUBLE,
					proc_right_rank, 2, MPI_COMM_WORLD, &send_right_req);
			handle_rc(rc, "error sending data to the right.");

			rc = MPI_Irecv(&(b[last_row][0]), dimensions, MPI_DOUBLE,
					proc_right_rank, 1, MPI_COMM_WORLD, &recv_right_req);
			handle_rc(rc, "error receiving data from the right.");
		}

		// reduce and swap array during async send and receive

		// bitwise and of local_done to find global_done
		// local_done will be 0 if a processor is not done
		// global_done will only be 1, if all processors are done
		rc = MPI_Allreduce(&local_done, &global_done, 1,
				MPI_CHAR, MPI_BAND, MPI_COMM_WORLD);
		handle_rc(rc, "error reducing done values.");

		// a now points to results, ready for next iteration
		swap_array(&a, &b);

		// wait for sends and receives before continuing on to next loop
		if (my_rank != 0)
		{
			rc = MPI_Wait(&send_left_req, MPI_STATUS_IGNORE);
			handle_rc(rc, "error waiting send left.");

			rc = MPI_Wait(&recv_left_req, MPI_STATUS_IGNORE);
			handle_rc(rc, "error waiting receive left.");
		}

		if (my_rank != processors - 1)
		{
			rc = MPI_Wait(&send_right_req, MPI_STATUS_IGNORE);
			handle_rc(rc, "error waiting send right.");

			rc = MPI_Wait(&recv_right_req, MPI_STATUS_IGNORE);
			handle_rc(rc, "error waiting receive right.");
		}
	}

	// gather results from each processor's 'a' array
	// into the root processors 'a' array
	// row 0 is border and only used for calculations, results start from row 1
	rc = MPI_Gatherv(&(a[1][0]), my_recv_count, MPI_DOUBLE,
			&(a[0][0]), recv_counts, recv_displs, MPI_DOUBLE, root,
			MPI_COMM_WORLD);
	handle_rc(rc, "error gathering data.");

	if (my_rank == root)
	{
		printf("iterations: %d\n", iterations);
	}
}

void alloc_work(int dimensions, int processors, int my_rank, int root,
		int *my_send_count, int *my_recv_count,
		int *send_counts, int *send_displs,
		int *recv_counts, int *recv_displs)
{
	// each processor will relax n rows
	int nrows = (dimensions - 2) / processors;

	// first m processors will relax nrows + 1 rows, where m is extra_rows
	int extra_rows = (dimensions - 2) % processors;

	// first m rows will relax n + 1 rows
	// multiply by dimensions to get total number of cells
	*my_recv_count = (nrows + (my_rank < extra_rows)) * dimensions;

	// number of cells to send, includes row above and row below
	*my_send_count = *my_recv_count + 2 * dimensions;

	// first processor will start at row 1, after n (= dimensions) cells
	int displacement = dimensions;

	if (my_rank == root)
	{
		for (int rank = 0; rank < processors; rank++)
		{
			recv_counts[rank] = (nrows + (rank < extra_rows)) * dimensions;
			send_counts[rank] = recv_counts[rank] + 2 * dimensions;

			// calculate start position, displacement relative to start
			recv_displs[rank] = displacement;
			displacement += recv_counts[rank];

			send_displs[rank] = recv_displs[rank] - dimensions;
		}
	}
}

void alloc_memory(int dimensions, int processors, int my_rank, int root,
		double ***a, double ***b, double **a_buf, double **b_buf,
		int **send_counts, int **send_displs,
		int **recv_counts, int **recv_displs)
{

	// only root needs space for whole array
	// rest only need space for the rows they are relaxing
	int rows = dimensions;
	if (my_rank != root)
	{
		// each processor will relax n rows,
		// first m processors will relax n + 1 rows,
		// + 2 extra for row above and below for relaxation calculations
		int nrows = (dimensions - 2) / processors;
		int extra_rows = (dimensions - 2) % processors;
		rows = nrows + (my_rank < extra_rows) + 2;
	}

	*a = malloc((unsigned long)rows * sizeof(double *));
	*b = malloc((unsigned long)rows * sizeof(double *));

	*a_buf = calloc((unsigned long)(rows * dimensions), sizeof(double));
	*b_buf = calloc((unsigned long)(rows * dimensions), sizeof(double));

	// only root requires send/recv counts of all sections
	// rest only require the send/recv count for their section
	if (my_rank == root)
	{
		*send_counts = malloc((unsigned long)processors * sizeof(int));
		*send_displs = malloc((unsigned long)processors * sizeof(int));
		*recv_counts = malloc((unsigned long)processors * sizeof(int));
		*recv_displs = malloc((unsigned long)processors * sizeof(int));
	}

	if (*a == NULL || *b == NULL || *a_buf == NULL || *b_buf == NULL
			|| send_counts == NULL || send_displs == NULL
			|| recv_counts == NULL || recv_displs == NULL)
	{
		printf("malloc failed.\n");
		exit(EXIT_FAILURE);
	}

	// each a[i] points to start of a row
	for (int i = 0; i < rows; i++)
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
	int rc, my_rank, root = 0, processors, dimensions;
	double precision;

	// init
	rc = MPI_Init(&argc, &argv);
	handle_rc(rc, "error starting MPI program.");

	// get rank
	rc = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	handle_rc(rc, "error retrieving rank.");

	// get number of processors
	rc = MPI_Comm_size(MPI_COMM_WORLD, &processors);
	handle_rc(rc, "error retrieving size of communicator.");

	process_args(argc, argv, &dimensions, &precision);

	if (my_rank == root)
	{
		printf("using dimension: %d\n", dimensions);
		printf("using processors: %d\n", processors);
		printf("using precision: %lf\n", precision);
	}

	double **a = NULL, **b = NULL, *a_buf = NULL, *b_buf = NULL;
	int *send_counts = NULL, *send_displs = NULL,
		*recv_counts = NULL, *recv_displs = NULL;

	alloc_memory(dimensions, processors, my_rank, root, &a, &b, &a_buf, &b_buf,
			&send_counts, &send_displs, &recv_counts, &recv_displs);

	int my_send_count, my_recv_count;
	alloc_work(dimensions, processors, my_rank, root,
			&my_send_count, &my_recv_count,
			send_counts, send_displs, recv_counts, recv_displs);

	if (my_rank == root)
	{
		populate_array(a, b, dimensions);
#ifdef DEBUG
		print_array(a, dimensions);
#endif
	}

	relax_array(a, b, my_rank, root, dimensions, processors, precision,
			my_send_count, my_recv_count,
			send_counts, send_displs, recv_counts, recv_displs);

	if (my_rank == root)
	{
#ifdef DEBUG
		print_array(a, dimensions);
#endif
	}

	dealloc_memory(a, b, a_buf, b_buf,
			send_counts, send_displs, recv_counts, recv_displs);

	rc = MPI_Finalize();
	handle_rc(rc, "error closing MPI program.");

	exit(EXIT_SUCCESS);
}
