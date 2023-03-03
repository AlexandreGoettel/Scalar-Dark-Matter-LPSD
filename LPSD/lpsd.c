/********************************************************************************
    lpsd.c
			  
    2003, 2004 by Michael Troebs, mt@lzh.de and Gerhard Heinzel, ghh@mpq.mpg.de

    calculate spectra from time series using discrete Fourier 
    transforms at frequencies equally spaced on a logarithmic axis
    
    lpsd does everything except user interface and data output
    
 ********************************************************************************/
#define SINCOS

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <fftw3.h>
#include "hdf5.h"

#include "config.h"
#include "ask.h"
#include "IO.h"
#include "genwin.h"
#include "debug.h"
#include "lpsd.h"
#include "misc.h"
#include "errors.h"

/*
20.03.2004: http://www.caddr.com/macho/archives/iolanguage/2003-9/549.html
*/
#ifndef __linux__
#include <windows.h>
struct timezone
{
  int tz_minuteswest;
  int tz_dsttime;
};
void
gettimeofday (struct timeval *tv, struct timezone *tz
	      __attribute__ ((unused)))
{
  long int count = GetTickCount ();
  tv->tv_sec = (int) (count / 1000);
  tv->tv_usec = (count % 1000) * 1000;
}

#else
#include <sys/time.h>		/* gettimeofday, timeval */
#endif

#ifdef __linux__
extern double round (double x);
/*
gcc (GCC) 3.3.1 (SuSE Linux) gives warning: implicit declaration of function `round'
without this line - math.h defines function round, but its declaration seems to be
missing in math.h 
*/

#else
/*
int round (double x) {
  return ((int) (floor (x + 0.5)));
}
*/
#endif

/********************************************************************************
 * 	global variables						   	
 ********************************************************************************/
static int nread;
static double winsum;
static double winsum2;
static double nenbw;		/* normalized equivalent noise bandwidth */
static double *dwin;		/* pointer to window function for FFT */

/********************************************************************************
 * 	functions								
 ********************************************************************************/


// Get mean over N first values of int sequence
double
get_mean (int* values, int N) {
    double _sum = 0;
    register int i;
    for (i = 0; i < N; i++) {
        _sum += values[i];
    }
    return _sum / N;
}


// @brief recursive function to count set bits
int
count_set_bits (int n)
{
    // base case
    if (n == 0)
        return 0;
    else
        // if last bit set add 1 else add 0
        return (n & 1) + count_set_bits(n >> 1);
}


int
get_next_power_of_two (int n)
{
    int output = n;
    if (!(count_set_bits(n) == 1 || n == 0))
      output = (int) pow(2, (int) log2(n) + 1);
    return output;
}


void
stride_over_array (double *data, int N, int stride, int offset, double *output)
{
    for (int i = offset; i < N; i += stride) *(output++) = data[i];
}


// Get the segment length as a function of the frequency bin j
// Rounded to nearest integer.
// TODO: replace with call to nffts?
int
get_N_j (int j, double fsamp, double fmin, double fmax, int Jdes) {
    double g = log(fmax) - log(fmin);  // TODO: could consider making g part of cfg
    return round (fsamp/fmin * exp(-j*g / (Jdes - 1.)) / (exp(g / (Jdes - 1.)) - 1.));
}

// Get the frequency of bin j
// TODO: replace with call to fspec?
double
get_f_j (int j, double fmin, double fmax, int Jdes) {
    double g = log(fmax) - log(fmin);  // TODO: could consider making g part of cfg
    return fmin*exp(j*g / (Jdes - 1.));
}

void
fill_ordered_coefficients(int n, int *coefficients) {
    coefficients[0] = 0;
    coefficients[1] = 1;
    if (n == 1) return;
    for (int m = 2; m <= n; m++) {
        int two_to_m = pow(2, m);
        int two_to_m_minus_one = pow(2, m-1);
        int tmp[two_to_m];
        for (int i = 0; i < two_to_m_minus_one; i++) {
            tmp[2*i] = coefficients[i];
            tmp[2*i+1] = coefficients[i] + two_to_m_minus_one;
        }
        for (int i = 0; i < two_to_m; i++) coefficients[i] = tmp[i];
    }
}


static void
getDFT2 (int nfft, double bin, double fsamp, double ovlp, double *rslt,
         int *avg, struct hdf5_contents *contents)
{
  /* Configure variables for DFT */
  int max_samples_in_memory = 5*6577770;  // Around 500 MB //TODO: this shouldn't be hard-coded!
//  int max_samples_in_memory = 512;  // tmp
  if (max_samples_in_memory > nfft) max_samples_in_memory = nfft; // Don't allocate more than you need

  /* Allocate data and window memory segments */
  double *strain_data_segment = (double*) xmalloc(max_samples_in_memory * sizeof(double));
  double *window = (double*) xmalloc(2*max_samples_in_memory * sizeof(double));
  assert(window != 0 && strain_data_segment != 0);

  //////////////////////////////////////////////////
  /* Calculate DFT over separate memory windows */
  int window_offset, count;
  int memory_unit_index = 0;
  int remaining_samples = nfft;
  int nsum = floor(1+(nread - nfft) / floor(nfft * (1.0 - (double) (ovlp / 100.))));
  int tmp = (nsum-1)*floor(nfft * (1.0 - (double) (ovlp / 100.)))+nfft;
  if (tmp == nread) nsum--;  /* Adjust for edge case */

  double dft_results[2*nsum];  /* Real and imaginary parts of DFTs */
  memset(dft_results, 0, 2*nsum*sizeof(double));

  while (remaining_samples > 0)
  {
    if (remaining_samples > max_samples_in_memory)
    {
      count = max_samples_in_memory;
      remaining_samples -= max_samples_in_memory;
    } else {
      count = remaining_samples;
      remaining_samples = 0;
    }
    window_offset = memory_unit_index * max_samples_in_memory;
    memory_unit_index++;

    // Calculate window
    makewinsincos_indexed(nfft, bin, window, &winsum, &winsum2, &nenbw,
                          window_offset, count, window_offset == 0);

    // Loop over data segments
    int start = 0;
    register int _nsum = 0;
    hsize_t data_count[1] = {count};
    hsize_t data_rank = 1;
    while (start + nfft < nread)
    {
      // Load data
      hsize_t data_offset[1] = {start + window_offset};
      read_from_dataset(contents, data_offset, data_count, data_rank, data_count, strain_data_segment);

      // Calculate DFT
      register int i;
      for (i = 0; i < count; i++)
      {
        dft_results[_nsum*2] += window[i*2] * strain_data_segment[i];
        dft_results[_nsum*2 + 1] += window[i*2 + 1] * strain_data_segment[i];
      }
      start += nfft * (1.0 - (double) (ovlp / 100.));  /* go to next segment */
      _nsum++;
    }
  }

  /* Sum over dft_results to get total */
  register int i;
  double total = 0;  /* Running sum of DFTs */
  for (i = 0; i < nsum; i++)
  {
    total += dft_results[i*2]*dft_results[i*2] + dft_results[i*2+1]*dft_results[i*2+1];
  }
  //////////////////////////////////////////////////

  /* Return result */
  rslt[0] = total / nsum;
  
  /* This sets the variance to zero. This is not true, but we are not using the variance. */
  rslt[1] = 0;
  
  rslt[2] = rslt[0];
  rslt[3] = rslt[1];
  rslt[0] *= 2. / (fsamp * winsum2);	/* power spectral density */
  rslt[1] *= 2. / (fsamp * winsum2);	/* variance of power spectral density */
  rslt[2] *= 2. / (winsum * winsum);	/* power spectrum */
  rslt[3] *= 2. / (winsum * winsum);	/* variance of power spectrum */

  *avg = nsum;

  /* clean up */
  xfree(window);
  xfree(strain_data_segment);
}


/*
	calculates paramaters for DFTs
	output
		fspec		frequencies in spectrum
		bins		bins for DFTs
		nffts		dimensions for DFTs
 ********************************************************************************
 	Naming convention	source code	publication
        i		    j
        fres	    r''
        ndft	    L(j)
        bin		    m(j)
 */
static void
calc_params (tCFG * cfg, tDATA * data)
{
  double fres, f, bin, g;
  int i, i0, ndft;

  g = log ((*cfg).fmax / (*cfg).fmin);
  i = (*cfg).nspec * (*cfg).iter;
  i0 = i;
  f = (*cfg).fmin * exp (i * g / ((*cfg).Jdes - 1.));
  while (f <= (*cfg).fmax && i / (*cfg).nspec < (*cfg).iter + 1)
   {
      fres = f * (exp (g / ((*cfg).Jdes - 1.)) - 1);
      ndft = round ((*cfg).fsamp / fres);
      bin = (f / fres);
      (*data).fspec[i - i0] = f;
      (*data).nffts[i - i0] = ndft;
      (*data).bins[i - i0] = bin;
      i++;
      f = (*cfg).fmin * exp (i * g / ((*cfg).Jdes - 1.));
  }
  (*cfg).nspec = i - i0;
  (*cfg).fmin = (*data).fspec[0];
  (*cfg).fmax = (*data).fspec[(*cfg).nspec - 1];
}

void
calculate_lpsd (tCFG * cfg, tDATA * data)
{
  int k;			/* 0..nspec */
  int k_start = 0;		/* N. lines in save file. Post fail start point */
  char ch;			/* For scanning through checkpointing file */
  int Nsave = (*cfg).nspec / 100; /* Frequency of data checkpointing */
  int j; 			/* Iteration variables for checkpointing data */
  FILE * file1;			/* Output file, temp for checkpointing */
  double rslt[4];		/* rslt[0]=PSD, rslt[1]=variance(PSD) rslt[2]=PS rslt[3]=variance(PS) */
  double progress;

  struct timeval tv;
  double start, now, print;

  /* Check output file for saved checkpoint */
  file1 = fopen((*cfg).ofn, "r");
  if (file1){
      while((ch=fgetc(file1)) != EOF){
          if(ch == '\n'){
              k_start++;
          }
      }
  fclose(file1);
  printf("Backup collected. Starting from k = %i\n", k_start);
  }
  else{
      printf("No backup file. Starting from fmin\n");
      k_start = 0;
  }
  printf ("Checkpointing every %i iterations\n", Nsave);
  printf ("Computing output:  00.0%%");
  fflush (stdout);
  gettimeofday (&tv, NULL);
  start = tv.tv_sec + tv.tv_usec / 1e6;
  now = start;
  print = start;
  
  /* Start calculation of LPSD from saved checkpoint or zero */
  struct hdf5_contents *contents = read_hdf5_file((*cfg).ifn, (*cfg).dataset_name);
  for (k = k_start; k < (*cfg).nspec; k++)
    {
      getDFT2((*data).nffts[k], (*data).bins[k], (*cfg).fsamp, (*cfg).ovlp,
	          &rslt[0], &(*data).avg[k], contents);

      (*data).psd[k] = rslt[0];
      (*data).varpsd[k] = rslt[1];
      (*data).ps[k] = rslt[2];
      (*data).varps[k] = rslt[3];
      gettimeofday (&tv, NULL);
      now = tv.tv_sec + tv.tv_usec / 1e6;
      if (now - print > PSTEP)
	{
	  print = now;
	  progress = (100 * ((double) k)) / ((double) ((*cfg).nspec));
	  printf ("\b\b\b\b\b\b%5.1f%%", progress);
	  fflush (stdout);
	}

      /* If k is a multiple of Nsave then write data to backup file */
      if(k % Nsave  == 0 && k != k_start){
          file1 = fopen((*cfg).ofn, "a");
          for(j=k-Nsave; j<k; j++){
		fprintf(file1, "%e	", (*data).psd[j]);
		fprintf(file1, "%e	", (*data).ps[j]);
		fprintf(file1, "%d	", (*data).avg[j]);
		fprintf(file1, "\n");
          }
          fclose(file1);
      }
      else if(k == (*cfg).nspec - 1){
          file1 = fopen((*cfg).ofn, "a");
          for(j=Nsave*(k/Nsave); j<(*cfg).nspec; j++){
		fprintf(file1, "%e	", (*data).psd[j]);
		fprintf(file1, "%e	", (*data).ps[j]);
		fprintf(file1, "%d	", (*data).avg[j]);
		fprintf(file1, "\n");
          }
          fclose(file1);
      }
    }
  /* finish */
  close_hdf5_contents(contents);
  printf ("\b\b\b\b\b\b  100%%\n");
  fflush (stdout);
  gettimeofday (&tv, NULL);
  printf ("Duration (s)=%5.3f\n\n", tv.tv_sec - start + tv.tv_usec / 1e6);
}


// @brief Calculate FFT on data of length N
// @brief This implementation puts everything in memory, serves as test
// @brief Takes in real data
// @param Custom bin number, necessary for logarithmic frequency spacing
// @brief Memory contents reach 3N in main loop (+ 2N from recursion)
// TODO: implement Bergland's algorithm
// TODO: sin/cos optimisation
void
FFT(double *data_real, double *data_imag, int N,
    double *output_real, double *output_imag)
{
    if (N == 1) {
        output_real[0] = data_real[0];
        output_imag[0] = data_imag[0];
        return;
    }
    int m = N / 2;

    // Separate even part in real/imaginary
    double *x_even_real = (double*) xmalloc(m*sizeof(double));
    double *x_even_imag = (double*) xmalloc(m*sizeof(double));
    stride_over_array(data_imag, N, 2, 0, x_even_imag);
    stride_over_array(data_real, N, 2, 0, x_even_real);
    // Calculate FFT over halved arrays
    double *X_even_real = (double*) xmalloc(m*sizeof(double));
    double *X_even_imag = (double*) xmalloc(m*sizeof(double));
    FFT(x_even_real, x_even_imag, m, X_even_real, X_even_imag);
    // Clean up
    xfree(x_even_real);
    xfree(x_even_imag);

    // Repeat for odd part
    double *x_odd_real = (double*) xmalloc(m*sizeof(double));
    double *x_odd_imag = (double*) xmalloc(m*sizeof(double));
    stride_over_array(data_real, N, 2, 1, x_odd_real);
    stride_over_array(data_imag, N, 2, 1, x_odd_imag);
    // Calculate FFT over halved arrays
    double *X_odd_real = (double*) xmalloc(m*sizeof(double));
    double *X_odd_imag = (double*) xmalloc(m*sizeof(double));
    FFT(x_odd_real, x_odd_imag, m, X_odd_real, X_odd_imag);
    // Clean up
    xfree(x_odd_real);
    xfree(x_odd_imag);

    // Calculate exponential term to multiply to X_odd
    double exp_factor = 2.0 * M_PI / ((double) N);
    for (int i = 0; i < m; i++) {
        double y = cos(i*exp_factor);
        double x = -sin(i*exp_factor);
        double b = X_odd_real[i];
        double a = X_odd_imag[i];
        X_odd_real[i] = b*y - a*x;
        X_odd_imag[i] = a*y + b*x;
    }

    // Calculate final answer
    for (int i = 0; i < m; i++) {
        output_real[i] = X_even_real[i] + X_odd_real[i];
        output_imag[i] = X_even_imag[i] + X_odd_imag[i];
        output_real[i+m] = X_even_real[i] - X_odd_real[i];
        output_imag[i+m] = X_even_imag[i] - X_odd_imag[i];
    }

    // Clean up
    xfree(X_even_real);
    xfree(X_odd_real);
    xfree(X_even_imag);
    xfree(X_odd_imag);
}


// Perform an FFT while controlling how much gets in memory by manually calculating the
// top layers of the pyramid over sums
void
FFT_control_memory(int Nj0, int Nfft, int Nmax,
                   struct hdf5_contents *contents, struct hdf5_contents *_contents,
                   double *winsum, double *winsum2, double *nenbw)
{
    // Determine manual recursion depth
    // Nfft and Nmax must be powers of two!!
    int n_depth = round(log2(Nfft) - log2(Nmax));  // use round() to avoid float precision trouble

    // Get 2^n_depth data samples, then iteratively work down to n = 1
    int two_to_n_depth = pow(2, n_depth);
    int Nj0_over_two_n_depth = Nj0 / two_to_n_depth + 1;
    int ordered_coefficients[two_to_n_depth];
    fill_ordered_coefficients(n_depth, ordered_coefficients);

    // Approx (5 * 16 * Nmax) bits in memory
    double *data_subset_real = (double*)xmalloc(Nmax*sizeof(double));
    double *data_subset_imag = (double*)xmalloc(Nmax*sizeof(double));
    memset(data_subset_imag, 0, Nmax*sizeof(double));
    double *fft_output_real = (double*)xmalloc(Nmax*sizeof(double));
    double *fft_output_imag = (double*)xmalloc(Nmax*sizeof(double));
    double *window_subset = (double*)xmalloc(Nj0_over_two_n_depth*sizeof(double));

    // Perform FFTs on bottom layer of pyramid and save results to temporary file
    for (int i = 0; i < two_to_n_depth; i++) {
        // Read data
        hsize_t count[1] = {Nj0_over_two_n_depth};
        hsize_t offset[1] = {ordered_coefficients[i]};
        hsize_t stride[1] = {two_to_n_depth};
        hsize_t rank = 1;
        read_from_dataset_stride(contents, offset, count, stride, rank, count, data_subset_real);

        // Zero-pad data
        for (int j = Nj0_over_two_n_depth; j < Nmax; j++) data_subset_real[j] = 0;

        // Generate window
        makewin_indexed(Nj0, ordered_coefficients[i], two_to_n_depth, window_subset,
                        winsum, winsum2, nenbw, i == 0);

        // Piecewise multiply
        for (int j = 0; j < Nj0_over_two_n_depth; j++) data_subset_real[j] *= window_subset[j];

        // Take FFT
        FFT(data_subset_real, data_subset_imag, Nmax, fft_output_real, fft_output_imag);

        // Save real part to file
        hsize_t _offset[2] = {0, i*Nmax};
        hsize_t _count[2] = {1, Nmax};
        hsize_t _data_rank = 1;
        hsize_t _data_count[1] = {Nmax};
        write_to_hdf5(_contents, fft_output_real, _offset, _count, _data_rank, _data_count);
        // Save imaginary part
        _offset[0] = 1;
        write_to_hdf5(_contents, fft_output_imag, _offset, _count, _data_rank, _data_count);
        // Note: if I saved real/imag in one 2D array instead of two arrays,
        // I would only need one call to write_to_hdf5 in this loop. Could save time?
    }
    // winsums are correct here
    // Clean-up
    xfree(data_subset_real);
    xfree(data_subset_imag);
    xfree(window_subset);
    xfree(fft_output_real);
    xfree(fft_output_imag);

    // TODO: don't need to write the last iteration of the pyramid to file as I could work with it here directly, small speed-up
    // Put 5 * 16 * Nmax bits in memory
    double *even_terms_real = (double*)xmalloc(Nmax*sizeof(double));
    double *even_terms_imag = (double*)xmalloc(Nmax*sizeof(double));
    double *odd_terms_real = (double*)xmalloc(Nmax*sizeof(double));
    double *odd_terms_imag = (double*)xmalloc(Nmax*sizeof(double));
    double *write_vector = (double*)xmalloc(Nmax*sizeof(double));
    // Now loop over the rest of the pyramid
    while (n_depth > 0) {
        // Iterate n_depth
        n_depth--;
        two_to_n_depth = pow(2, n_depth);
        Nj0_over_two_n_depth = Nj0 / two_to_n_depth;
        int m = pow(2, (int)round(log2(Nfft) - log2(Nmax) - n_depth - 1));  //round() to avoid float precision

        // Loop over data
        for (int i = 0; i < two_to_n_depth; i++) {
            // Loop over memory units (of length Nmax)
            for (int j = 0; j < m; j++) {
                // Load even terms
                hsize_t offset[2] = {0, j*Nmax};
                hsize_t count[2] = {1, Nmax};
                hsize_t data_rank = 1;
                hsize_t data_count[1] = {Nmax};
                read_from_dataset(_contents, offset, count, data_rank, data_count, even_terms_real);
                offset[0] = 1;
                read_from_dataset(_contents, offset, count, data_rank, data_count, even_terms_imag);

                // Load odd terms
                offset[1] = (j+1)*Nmax;
                read_from_dataset(_contents, offset, count, data_rank, data_count, odd_terms_imag);
                offset[0] = 0;
                read_from_dataset(_contents, offset, count, data_rank, data_count, odd_terms_real);

                // Piecewise (complex) multiply odd terms with exp term
                double exp_factor = 2.0 * M_PI / ((double) Nfft / two_to_n_depth);
                for (int k = 0; k < Nmax; k++) {
                    double y = cos((j*Nmax+k)*exp_factor);
                    double x = -sin((j*Nmax+k)*exp_factor);
                    double a = odd_terms_imag[k];
                    double b = odd_terms_real[k];
                    odd_terms_real[k] = b*y - a*x;
                    odd_terms_imag[k] = a*y + b*x;
                }

                // Combine left side
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_real[k] + odd_terms_real[k];
                hsize_t offset_left[2] = {0, j*Nmax};
                write_to_hdf5(_contents, write_vector, offset, count, data_rank, data_count);
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_imag[k] + odd_terms_imag[k];
                offset[0] = 1;
                write_to_hdf5(_contents, write_vector, offset, count, data_rank, data_count);

                // Combine right side
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_real[k] - odd_terms_real[k];
                hsize_t offset_right[2] = {(int)Nfft/pow(2, n_depth+1)+j*Nmax, (int)Nfft/pow(2, n_depth+1)+j*Nmax};
                offset[0] = 0;
                write_to_hdf5(_contents, write_vector, offset, count, data_rank, data_count);
                for (int k = 0; k < Nmax; k++)
                    write_vector[k] = even_terms_imag[k] - odd_terms_imag[k];
                offset[0] = 1;
                write_to_hdf5(_contents, write_vector, offset, count, data_rank, data_count);
            }
            // loop(read[j*M:(j+1)*M], combine[j*M:(j+1)*M], combine[(j+pow^(mem-1)*M:(j+(pow^mem-1)+1)*M];
        }
    }
    // Clean up
    xfree(even_terms_real);
    xfree(even_terms_imag);
    xfree(odd_terms_real);
    xfree(odd_terms_imag);
    xfree(write_vector);
}


// @brief Use const. N approximation for a given epsilon
void
calculate_fft_approx (tCFG * cfg, tDATA * data)
{
    // Track time and progress
    struct timeval tv;
    printf ("Computing output:  00.0%%");
    fflush (stdout);
    gettimeofday (&tv, NULL);
    double start = tv.tv_sec + tv.tv_usec / 1e6;
    double now, print, progress;
    print = start;
    now = start;

    // Define variables
    double epsilon = 0.1;  // TODO: pass arg
    double g = log(cfg->fmax / cfg->fmin);

    // Prepare data file
    struct hdf5_contents *contents = read_hdf5_file((*cfg).ifn, (*cfg).dataset_name);

    // Loop over blocks
    register int i;
    int j, j0;
    j = j0 = 0;

    while (true) {
        // Get index of the end of the block - the frequency at which the approximation is valid up to epsilon
        j0 = j;
        int Nj0 = get_N_j(j0, cfg->fsamp, cfg->fmin, cfg->fmax, cfg->Jdes);
        j = - (cfg->Jdes - 1.) / g * log(Nj0*(1. - epsilon) * cfg->fmin/cfg->fsamp * (exp(g / (cfg->Jdes - 1.)) - 1.));
//        printf("j, Jdes: %d, %d\n", j, cfg->Jdes);
        if (j >= cfg->Jdes) break; // TODO: take care of edge case

        // Prepare segment loop
        int delta_segment = floor(Nj0 * (1.0 - (double) (cfg->ovlp / 100.)));
        int n_segments = floor(1 + (nread - Nj0) / delta_segment);
        /* Adjust for edge case */
        int tmp = (n_segments - 1)*delta_segment + Nj0;
        if (tmp == nread) n_segments--;
        double total[j - j0];  // TODO: malloc?
        memset(total, 0, (j-j0)*sizeof(double));

        // Prepare FFT
        int Nfft = get_next_power_of_two(Nj0);
        // int max_samples_in_memory = 33554432;  // 2^25 b = 0.5 Gb if double  // TODO: pass arg
        int max_samples_in_memory = 131072;  // 2^17, for testing #deleteme
        // Open temporary hdf5 file to temporarily store information to disk in the loop
        hsize_t rank = 2;  // real + imaginary
        hsize_t dims[2] = {2, Nfft};
//        printf("Nj0, Nfft, Nmax: %d, %d, %d\n", Nj0, Nfft, max_samples_in_memory);
        struct hdf5_contents *_contents = open_hdf5_file("tmp.h5", "fft_contents", rank, dims);

        // Loop over segments
        register int i_segment, ji;
        for (i_segment = 0; i_segment < n_segments; i_segment++){
            // Take FFT of data * window
            // winsum and winsum2 are calculated in here
            if (Nfft > max_samples_in_memory)
                FFT_control_memory(Nj0, Nfft, max_samples_in_memory, contents, _contents,
                                   &winsum, &winsum2, &nenbw);
            else
                // TODO: Don't use save to file when Nfft <= max_samples_in_memory!

            // Load results that are between j0 and j
            int jfft_min = floor(Nfft * cfg->fmin/cfg->fsamp * exp(j0*g/(cfg->Jdes - 1.)));
            int jfft_max = ceil(Nfft * cfg->fmin/cfg->fsamp * exp(j*g/(cfg->Jdes - 1.)));
            double fft_real[jfft_max - jfft_min], fft_imag[jfft_max - jfft_min];  // TODO: malloc?
            hsize_t count[2] = {1, jfft_max - jfft_min};
            hsize_t offset[2] = {0, jfft_min};
            hsize_t data_rank = 1;
            hsize_t data_count[1] = {count[1]};
            read_from_dataset(_contents, offset, count, data_rank, data_count, fft_real);
            offset[0] = 1;
            read_from_dataset(_contents, offset, count, data_rank, data_count, fft_imag);

            // Interpolate results
            for (ji = j0; ji < j; ji++) {
                int jfft = floor(Nfft * cfg->fmin/cfg->fsamp * exp(ji*g/(cfg->Jdes - 1.)));
                double x = get_f_j(ji, cfg->fmin, cfg->fmax, cfg->Jdes);
                double y1 = fft_real[jfft-jfft_min]*fft_real[jfft-jfft_min] + fft_imag[jfft-jfft_min]*fft_imag[jfft-jfft_min];
                double y2 = fft_real[jfft-jfft_min+1]*fft_real[jfft-jfft_min+1] + fft_imag[jfft-jfft_min+1]*fft_imag[jfft-jfft_min+1];
                double x1 = cfg->fsamp / Nfft * jfft;
                double x2 = cfg->fsamp / Nfft * (jfft+1);
                total[ji - j0] += (y1*(x2 - x) - y2*(x1 - x)) / (x2 - x1);
            }
        }
        // Normalise results and add to data->psd and data->ps
        for (ji = 0; ji < j - j0; ji++) {
            data->psd[ji+j0] = total[ji] * 2. / (n_segments * cfg->fsamp * winsum2);
            data->ps[ji+j0] = total[ji] * 2 / (n_segments * winsum*winsum);
        }

        // Progress tracking
        progress = 100. * (double) j / cfg->Jdes;
        printf ("\b\b\b\b\b\b%5.1f%%", progress);
        fflush (stdout);

        // Clean-up
        close_hdf5_contents(_contents);
    }
}

/*
	works on cfg, data structures of the calling program
*/
void
calculateSpectrum (tCFG * cfg, tDATA * data)
{
  nread = floor (((*cfg).tmax - (*cfg).tmin) * (*cfg).fsamp + 1);

  calc_params (cfg, data);
  if ((*cfg).METHOD == 0) calculate_lpsd (cfg, data);
  else if ((*cfg).METHOD == 1) calculate_fft_approx (cfg, data);
  else gerror("Method not implemented.");
}
