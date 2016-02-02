#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <tbb/tbb.h>
#define KILO    (1024)
#define DEFAULT_SIZE_IN_KB (64)
#define NUM_ARGS  1
#define NUM_TIMERS  1

typedef double doubleType;
typedef struct {
  doubleType re;
  doubleType im;
} Complex;

void initialize(unsigned Size, Complex *a) {
  unsigned i;
  for(i = 0; i < Size; i++) {
    a[i].re = 1.0;
    a[i].im = 0.0;
  }
}

int test_array(unsigned Size, Complex *a) {
  register unsigned i;
  unsigned OK = 1;

  if((a[0].re == Size) && (a[0].im == 0)) {
    for(i = 1; i < Size; i++)
      if (a[i].re != 0.0 || a[i].im != 0.0) {
        OK = 0;
        break;
      }
  }
  else OK = 0;
  return OK;
}

void Roots(unsigned Size, Complex *W) {
  register unsigned i;
  double phi;
  Complex Omega;

  phi = 4 * atan(1.0) / (double)Size;                         /* PI/Size */
  Omega.re = cos(phi);
  Omega.im = sin(phi);
  W[0].re = 1.0;
  W[0].im = 0.0;
  for(i = 1; i < Size; i++) {
    W[i].re = W[i-1].re * Omega.re - W[i-1].im * Omega.im;
    W[i].im = W[i-1].re * Omega.im + W[i-1].im * Omega.re;
  }
}

unsigned get_params(int argc, char *argv[]) {
  char usage_str[] = "<size_in_Kb>";
  unsigned sizeInKb;

  if (argc == 2)
    sizeInKb = atoi(argv[1]);
  else
    if (argc == 1)
      sizeInKb = DEFAULT_SIZE_IN_KB;
    else {
      printf("\nUse: %s %s\n", argv[0], usage_str);
      exit(-1);
    }
  printf("\nUse: %s %s\n", argv[0], usage_str);
  printf("Running with Size: %d K\n", sizeInKb);
  return sizeInKb;
}
/* ---------------------------------------------------------------------- */

void FFT(Complex *A, Complex *a, Complex *W, unsigned N,unsigned stride, Complex *D);

class ApplySolver {
	Complex *A;
	Complex *a;
	Complex *W;
	unsigned n;
	unsigned stride;
	Complex *D;

	public:

	void operator()(const tbb::blocked_range<size_t>& r) const {
		for(int i = r.begin(), i_end = r.end(); i < i_end; i++){
			FFT(D + i * n, a + i * stride, W, n, stride << 1, A + i * n);
		}
  	}
	//Constructior
	ApplySolver(Complex *_A, Complex *_a, Complex *_W, unsigned _N,unsigned _stride, Complex *_D):
	A(_A),a(_a),W(_W),n(_N),stride(_stride),D(_D){}
};



class Calculator {
        Complex *A;
        Complex *W;
        unsigned n;
        unsigned stride;
	Complex *B;
	Complex *C;
	public:
        void operator()(const tbb::blocked_range<size_t>& r) const {
		Complex Aux, *pW;
                for(int i = r.begin(), i_end = r.end(); i < i_end; i++){                      
                      pW = W + i * stride;
                      Aux.re = pW->re * C[i].re - pW->im * C[i].im;
                      Aux.im = pW->re * C[i].im + pW->im * C[i].re;
                      A[i].re = B[i].re + Aux.re;
                      A[i].im = B[i].im + Aux.im;
                      A[i+n].re = B[i].re - Aux.re;
                      A[i+n].im = B[i].im - Aux.im;
                }
        }
        //Constructior
        Calculator(unsigned _stride,unsigned _n, Complex *_A,Complex *_B,Complex *_C,Complex *_W):
        stride(_stride),n(_n),A(_A),B(_B),C(_C),W(_W){}        
};

//-----------------------------------------------------


void FFT(Complex *A, Complex *a, Complex *W, unsigned N,unsigned stride, Complex *D) {
  Complex *B, *C;
  //Complex Aux, *pW;
  unsigned n;
  int i;

  if (N == 1) {
    A[0].re = a[0].re;
    A[0].im = a[0].im;
  }
  else {
	/* Division stage without copying input data */
	n = (N >> 1);   /* N = N div 2 */

    	/* Subproblems resolution stage */
	//static tbb::affinity_partitioner ap1;
	tbb::parallel_for(tbb::blocked_range<size_t>(0, 2),ApplySolver(A,a,W,n,stride,D)/*,ap1*/);
	
	/*tbb::parallel_for( tbb::blocked_range<int>(0, 1),
    		[&]( const tbb::blocked_range<int> &r ) {
		        for(int i = r.begin(), i_end = r.end(); i <= i_end; i++){                
	                FFT(D + i * n, a + i * stride, W, n, stride << 1, A + i * n);
        	}
	},ap1);*/

    /* Combination stage */

    B = D;
    C = D + n;

	static tbb::affinity_partitioner ap2;
	//tbb::parallel_for(tbb::blocked_range<size_t>(0, n),Calculator(stride, n, A, B, C, W),tbb::auto_partitioner());
	tbb::parallel_for( tbb::blocked_range<int>(0, n,1000),
	    [&]( const tbb::blocked_range<int> &r ) {
		//Complex Aux, *pW;		
        	for(int i = r.begin(), i_end = r.end(); i <= i_end-1; i++){
		      Complex Aux, *pW;
		      pW = W + i * stride;
		      Aux.re = pW->re * C[i].re - pW->im * C[i].im;
		      Aux.im = pW->re * C[i].im + pW->im * C[i].re;
		      A[i].re = B[i].re + Aux.re;
		      A[i].im = B[i].im + Aux.im;
		      A[i+n].re = B[i].re - Aux.re;
		      A[i+n].im = B[i].im - Aux.im;
		}
	},ap2);
  }
}

//-----------------------------------------------------

int main(int argc, char *argv[])
{
  unsigned N;
  Complex *a, *A, *W, *D;
  int NUMTHREADS;

  char *PARAM_NAMES[NUM_ARGS] = {"Size of the input signal (in Kb)"};
  char *TIMERS_NAMES[NUM_TIMERS] = {"Total_time" };
  char *DEFAULT_VALUES[NUM_ARGS] = {"64"};

  doubleType  start, finish;
  N = KILO * get_params(argc, argv);
  a = (Complex*)calloc(N, sizeof(Complex));
  A = (Complex*)calloc(N, sizeof(Complex));
  D = (Complex*)calloc(N, sizeof(Complex));
  W = (Complex*)calloc(N>>1, sizeof(Complex));

  if((a==NULL) || (A==NULL) || (D==NULL) || (W==NULL)) {
    printf("Not enough memory initializing arrays\n");
    exit(1);
  }

  initialize(N, a);
  Roots(N >> 1, W);/* Initialise the vector of imaginary roots */

  tbb::task_scheduler_init init;//(tbb::task_scheduler_init::deferred);

  //init.initialize(32);

  start = omp_get_wtime();
  FFT(A, a, W, N, 1, D);
  finish = omp_get_wtime();
  
  /* Display results and time */
  printf("Test array: ");
  if (test_array(N, A))
    printf("Ok\n");
  else
    printf("Fails\n");
  printf("TBB version time usage =%g\n",(finish-start));
 
  free(W);
  free(D);
  free(A);
  free(a);
}
