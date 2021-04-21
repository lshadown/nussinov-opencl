// Copyright (c) 2009-2011 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly

#ifndef __linux__
#include "stdafx.h"
#else
#include <string.h>
#endif

#define floord(n,d) floor(((double)(n))/((double)(d)))
#define ceild(n,d) ceil(((double)(n))/((double)(d)))

#include <iostream>
#include "basic.hpp"
#include "cmdparser.hpp"
#include "oclobject.hpp"
#include "utils.h"
#include "library.h"
#include <chrono>

#define N 1000

string method = "tilecor";


char* RNA = new char(N + 5);

typedef cl_int arrtype[N];


using namespace std;



void generateInput(cl_int* S, size_t width, size_t height)
{

    srand(12345);

    // random initialization of input
    for (cl_int i = 0; i <  width * height; ++i)
    {
 //       p_input[i] = (rand() | (rand()<<15) ) & 0xFFFFFF;
    }

    //p_input[0] = -2;
}

// inline functions for reference kernel
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif


// sekwencyjnie
void NussinovReference(cl_int S[N][N])
{
    int i, j, k;
    
    for (i = N - 1; i >= 0; i--) {
        for (j = i + 1; j < N; j++) {
            for (k = 0; k < j - i; k++) {
                S[i][j] = max(S[i][k + i] + S[k + i + 1][j], S[i][j]);
            }
            for (k = 0; k < 1; k++) {
                S[i][j] = max(S[i][j], S[i + 1][j - 1] + can_pair(RNA, i, j));

            }
        }
    }
}




// w OpenCL
float NussinovKernel(cl_int* S, cl_char* RNA_dev, cl_int width, cl_uint height, OpenCLBasic& ocl, OpenCLProgramOneKernel& executable, cl_int lb, cl_int ub, cl_int c1, cl_int* ST)
{
    double perf_start;
    double perf_stop;

   /* cl_int* my_lb = new cl_int[1];
    my_lb[0] = lb;
    cl_int* my_c1 = new cl_int[1];
    my_c1[0] = c1;
    */

    cl_int err = CL_SUCCESS;
    cl_uint numStages = 0;

    // allocate the buffer with some padding (to avoid boundaries checking)
    // CL_MEM_READ_WRITE
    cl_mem cl_S_buffer =
        clCreateBuffer
        (
            ocl.context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            zeroCopySizeAlignment(sizeof(cl_int) * width * (height)),
            S,
            &err
        );
    SAMPLE_CHECK_ERRORS(err);
    if (cl_S_buffer == (cl_mem)0)
        throw Error("Failed to create Input Buffer!");

    cl_mem cl_RNA_buffer =
        clCreateBuffer
        (
            ocl.context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            zeroCopySizeAlignment(sizeof(cl_char) * (width+5)),
            RNA_dev,
            &err
        );
    SAMPLE_CHECK_ERRORS(err);
    if (cl_RNA_buffer == (cl_mem)0)
        throw Error("Failed to create Input Buffer!");

    cl_mem cl_lb_buffer =
        clCreateBuffer
        (
            ocl.context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            zeroCopySizeAlignment(sizeof(cl_int)),
            &lb,
            &err
        );
    SAMPLE_CHECK_ERRORS(err);
    if (cl_lb_buffer == (cl_mem)0)
        throw Error("Failed to create Input Buffer!");

    cl_mem cl_c1_buffer =
        clCreateBuffer
        (
            ocl.context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            zeroCopySizeAlignment(sizeof(cl_int)),
            &c1,
            &err
        );
    SAMPLE_CHECK_ERRORS(err);
    if (cl_c1_buffer == (cl_mem)0)
        throw Error("Failed to create Input Buffer!");


    cl_mem cl_ST_buffer =
        clCreateBuffer
        (
            ocl.context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            zeroCopySizeAlignment(sizeof(cl_int) * width * (height)),
            ST,
            &err
        );
    SAMPLE_CHECK_ERRORS(err);


    err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), (void *) &cl_S_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 1, sizeof(cl_mem), (void*) &cl_RNA_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 2, sizeof(cl_mem), (void*) &cl_lb_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 3, sizeof(cl_mem), (void*) &cl_c1_buffer);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable.kernel, 4, sizeof(cl_mem), (void*)&cl_ST_buffer);
    SAMPLE_CHECK_ERRORS(err);

    size_t global_work_size[1] = {(size_t)ub-lb+1};


    // execute kernel

    err = clEnqueueNDRangeKernel(ocl.queue, executable.kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    // 2 wymiar, offset powinen byc NULL, global_work_size - ile pracowników w wymiarach

    SAMPLE_CHECK_ERRORS(err);
    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);



    void* tmp_ptr = NULL;
    tmp_ptr = clEnqueueMapBuffer(ocl.queue, cl_S_buffer, true, CL_MAP_READ, 0, sizeof(cl_int) * width * (height), 0, NULL, NULL, NULL);
    if(tmp_ptr!=S)
    {
        throw Error("clEnqueueMapBuffer failed to return original pointer");
    }


    err = clFinish(ocl.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clEnqueueUnmapMemObject(ocl.queue, cl_S_buffer, tmp_ptr, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseMemObject(cl_S_buffer);
    SAMPLE_CHECK_ERRORS(err);

    // retrieve perf. counter frequency
    return 0;
}




// main execution routine - performs median filtering with 3x3 kernel size
int main (int argc, const char** argv)
{
    //return code
    int ret = EXIT_SUCCESS;
    // pointer to the HOST buffers
    arrtype* S_seq = NULL;
    cl_int* S_par = NULL;
    cl_char* RNA_dev = NULL;

    cl_int* ST = NULL;

    

    try
    {
        // Define and parse command-line arguments.
        CmdParserCommon cmdparser(argc, argv);
        cmdparser.parse();


        int width = N;
        int height = N;
        cl_int c1, c0;



        // Create the necessary OpenCL objects up to device queue.
        OpenCLBasic oclobjects(
            cmdparser.platform.getValue(),
            cmdparser.device_type.getValue(),
            cmdparser.device.getValue()
        );

        

        printf("Input size is %d X %d\n", width, height);

        // allocate buffers 
        cl_int dev_alignment = zeroCopyPtrAlignment(oclobjects.device);
        size_t aligned_size = zeroCopySizeAlignment(sizeof(cl_int) * width * height, oclobjects.device);
        size_t aligned_size2 = zeroCopySizeAlignment(sizeof(cl_char) * (width+5), oclobjects.device);

        S_par = (cl_int*)aligned_malloc(aligned_size, dev_alignment);
        ST = (cl_int*)aligned_malloc(aligned_size, dev_alignment);
        RNA_dev = (cl_char*)aligned_malloc(aligned_size2, dev_alignment);
        S_seq = (arrtype*)malloc(sizeof(arrtype) * width);

        if(!(S_par && S_seq && ST))
        {
            throw Error("Could not allocate buffers on the HOST!");
        }


        // random input
        //generateInput(p_input, width, height);
        //float ocl_time = 0;

        printf("Executing OpenCL kernel...\n");
        auto start = std::chrono::steady_clock::now();
        cout << "Method: " << method << endl;

        if (method == "tilecor"){
            // Build kernel
            OpenCLProgramOneKernel executable(oclobjects, L"Nussinov.cl", "", "NussinovDeviceKernelTileCor");

            for (c1 = 1; c1 < N + floord((N - 2),128); c1 += 1)
            {
                cl_int ub = (c1 - 1) / 129;
                cl_int lb = max(0, -N + c1 + 1);
                //cout << ub - lb + 1 << endl;
                NussinovKernel(S_par, RNA_dev, width, height, oclobjects, executable, lb, ub, c1, ST);
            }
         }

        if (method == "tstile") {
            OpenCLProgramOneKernel executable(oclobjects, L"Nussinov.cl", "", "NussinovDeviceKernelTSTILE");
            for (c0 = 0; c0 <= floord(N - 2, 8); c0 += 1)
            {
                cout << c0 << endl;
                cl_int ub = min(c0, (N - 1) / 16);
                cl_int lb = (c0 + 1) / 2;
                if(ub - lb + 1 > 0)
                    NussinovKernel(S_par, RNA_dev, width, height, oclobjects, executable, lb, ub, c0, ST);
            }
        }

        if (method == "pluto") {
            OpenCLProgramOneKernel executable(oclobjects, L"Nussinov.cl", "", "NussinovDeviceKernelPluto");
            int t2;
            for (t2 = max(-1, ceild(-N - 13, 16)); t2 <= floord(N - 1, 16); t2++) {
                cl_int lbp = max(0, t2);
                cl_int ubp = min(floord(N - 1, 16), floord(16 * t2 + N + 13, 16));
                NussinovKernel(S_par, RNA_dev, width, height, oclobjects, executable, lbp, ubp, t2, ST);
            }
        }

        if (method == "li") {
            OpenCLProgramOneKernel executable(oclobjects, L"Nussinov.cl", "", "NussinovDeviceKernelLI");
            int diag;
            for (diag = 1; diag <= N - 1; diag++)
            {
                cl_int ub = N - diag - 1;
                cl_int lb = 0;
                NussinovKernel(S_par, RNA_dev, width, height, oclobjects, executable, lb, ub, diag, ST);
            }


        }
        if (method == "transpose") {
            OpenCLProgramOneKernel executable(oclobjects, L"Nussinov.cl", "", "NussinovDeviceKernelTranspose");

            int c0;

            for (c0 = 1; c0 < 2 * N - 2; c0 += 1) {
                //cout << c0 << endl;
                cl_int ub = min(N - 1, c0);
                cl_int lb = c0 / 2 + 1;
                NussinovKernel(S_par, RNA_dev, width, height, oclobjects, executable, lb, ub, c0, ST);
            }
        }
        if (method == "parallel") {
            OpenCLProgramOneKernel executable(oclobjects, L"Nussinov.cl", "", "NussinovDeviceKernelParallel");

            int i;


            for (i = N - 1; i >= 0; i--) {
                //cout << c0 << endl;
                cl_int ub = N-1;
                cl_int lb = i+1;
                if (ub - lb + 1 > 0)
                    NussinovKernel(S_par, RNA_dev, width, height, oclobjects, executable, lb, ub, i, ST);
            }
            
        }

 


        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

        printf("Executing reference...\n");
        start = std::chrono::steady_clock::now();
        NussinovReference(S_seq);
        end = std::chrono::steady_clock::now();
        elapsed_seconds = end - start;
        std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";


        printf("Performing verification...\n");
        bool    result = true;
        int     error_count = 0;
        for(int i = 0; i < height*width; i++)
        {

        }

        if(!result)
        {
            printf("ERROR: Verification failed.\n");
            ret = EXIT_FAILURE;
        }
        else
        {
            printf("Verification succeeded.\n");
        }

      //  printf("NDRange perf. counter time %f ms.\n", ocl_time*1000);

    }
    catch(const CmdParser::Error& error)
    {
        cerr
            << "[ ERROR ] In command line: " << error.what() << "\n"
            << "Run " << argv[0] << " -h for usage info.\n";
        ret = EXIT_FAILURE;
    }
    catch(const Error& error)
    {
        cerr << "[ ERROR ] Sample application specific error: " << error.what() << "\n";
        ret = EXIT_FAILURE;
    }
    catch(const exception& error)
    {
        cerr << "[ ERROR ] " << error.what() << "\n";
        ret = EXIT_FAILURE;
    }
    catch(...)
    {
        cerr << "[ ERROR ] Unknown/internal error happened.\n";
        ret = EXIT_FAILURE;
    }

    free(S_seq);
    aligned_free(S_par);


    return ret;
}

