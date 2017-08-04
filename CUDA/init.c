#include <stdio.h>
#include <cuda_runtime.h>
//打印函数
void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}
bool InitCUDA(){
    //获取设备数目
    int count;
    cudaGetDeviceCount(&count);
    if(count == 0){
        fprintf(stderr,"There is no device.\n");
        return false;
    }
    //获取设备属性
    int i;
    for(i = 0;i < count;++i){
        cudaGetDeviceProp prop;
        if(cudaGetDeviceProperties(&prop,i) == cudaSuccess){
            if(prop.major >= 1){
                printDeviceProp(prop);
                break;
            }
        }
    }
    if(i == count){
        fprintf(stderr,"There is no device supporting CUDA.\n");
        return false;
    }  
    //设置cuda设备
    cudaSetDevice(i);
}