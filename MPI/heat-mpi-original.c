/*
 * Iterative solver for heat distribution
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage( char *s )
{
    fprintf(stderr, 
	    "Usage: %s <input file> [result file]\n\n", s);
}

int main( int argc, char *argv[] )
{
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int locdim = (np*param.resolution)/numprocs

    if (myid == 0) {
      printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs-1);

    // algorithmic parameters
    algoparam_t param;
    int np;

    double runtime, flop;
    double residual=0.0;

    // check arguments
    if( argc < 2 )
    {
	usage( argv[0] );
	return 1;
    }

    // check input file
    if( !(infile=fopen(argv[1], "r"))  ) 
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
      
	usage(argv[0]);
	return 1;
    }

    // check result file
    resfilename= (argc>=3) ? argv[2]:"heat.ppm";

    if( !(resfile=fopen(resfilename, "w")) )
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for writing.\n\n", 
		resfilename);
	usage(argv[0]);
	return 1;
    }

    // check input
    if( !read_input(infile, &param) )
    {
	fprintf(stderr, "\nError: Error parsing input file.\n\n");
	usage(argv[0]);
	return 1;
    }
    print_params(&param);

    // set the visualization resolution
    
    param.u     = 0;
    param.uhelp = 0;
    param.uvis  = 0;
    param.visres = param.resolution;
   
    if( !initialize(&param) )
	{
	    fprintf(stderr, "Error in Solver initialization.\n\n");
	    usage(argv[0]);
            return 1;
	}

    // full size (param.resolution are only the inner points)
    np = param.resolution + 2;
    
    // starting time
    runtime = wtime();

    // send to workers the necessary data to perform computation
    for (int i=0; i<numprocs; i++) {
        if (i>0) {
                MPI_Send(&param.maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.resolution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.algorithm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.u[i*locdim], (np)*2 + locdim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.uhelp[i*locdim], (np)*2 + locdim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }
    }
    else {

    printf("I am worker %d and ready to receive work to do ...\n", myid);

    // receive information from master to perform computation locally

    int param.resolution, rows, np;
    int iter, param.maxiter;
    int param.algorithm;
    double residual;

    MPI_Recv(&param.maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&param.resolution, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&param.algorithm, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    rows = columns;
    np = columns + 2;

    // allocate memory for worker
    double * u = calloc( sizeof(double),(rows+2)*(columns+2) );
    double * uhelp = calloc( sizeof(double),(rows+2)*(columns+2) );
    if( (!u) || (!uhelp) )
    {
        fprintf(stderr, "Error: Cannot allocate memory\n");
        return 0;
    }
    
    // fill initial values for matrix with values received from master
    MPI_Recv(&param.u[0], np*2 + locdim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&param.uhelp[0], np*2 + locdim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    iter = 0;
    while(1) {
	switch( param.algorithm ) {
	    case 0: // JACOBI
	            residual = relax_jacobi(param.u, param.uhelp, para, np);
                 // sending last row to next thread
                    if( myid < numprocs-1)
                        MPI_Isend(&uhelp[locdim], np,MPI_DOUBLE,myid + 1,0,MPI_COMM_WORLD,&r[0]);
                    // sending first row to previous thread
                    MPI_Isend(&uhelp[np], np,MPI_DOUBLE,myid - 1,0,MPI_COMM_WORLD,&r[2]);

                //    MPI_Allreduce(&residual,&gresidual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

		    // Copy uhelp into u
		    for (int i=0; i<np; i++)
    		        for (int j=0; j<np; j++)
	    		    param.u[ i*np+j ] = param.uhelp[ i*np+j ];
		    
             // receiving first row from next thread
                   if (myid < numprocs-1)
                        MPI_Irecv(&u[locdim+np], np,MPI_DOUBLE,myid + 1,0,MPI_COMM_WORLD,&r[1]);
                    // receiving last row from previous thread
                    MPI_Recv(&u[0], np,MPI_DOUBLE,myid - 1,0,MPI_COMM_WORLD,&status);
                    MPI_Wait(&r[2],MPI_STATUS_IGNORE);

                    if( myid < numprocs-1)
                        MPI_Waitall(2,r,MPI_STATUS_IGNORE);
                
                break;

	    case 1: // RED-BLACK
		    residual = relax_redblack(param.u, np, np);
		    break;
	    case 2: // GAUSS
		    residual = relax_gauss(param.u, np, np);
		    break;
	    }

        iter++;

        // solution good enough ?
        if (residual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (param.maxiter>0 && iter>=param.maxiter) break;
    }

    // Flop count after iter iterations
    flop = iter * 11.0 * param.resolution * param.resolution;
    // stopping time
    runtime = wtime() - runtime;

    fprintf(stdout, "Time: %04.3f ", runtime);
    fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
	    flop/1000000000.0,
	    flop/runtime/1000000);
    fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

    // for plot...
    coarsen( param.u, np, np,
	     param.uvis, param.visres+2, param.visres+2 );
  
    write_image( resfile, param.uvis,  
		 param.visres+2, 
		 param.visres+2 );

    finalize( &param );

    fprintf(stdout, "Process %d finished computing with residual value = %f\n", myid, residual);

    MPI_Finalize();

    return 0;
