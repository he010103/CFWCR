The proposed tracker named CFWCR is implemented in Matlab. Our tracker has CPU and GPU version. The users can use demo_CFWCR.m to test. The vot integration is in the ./vot2017_trax. The GPU version code is ./vot2017_trax/CFWCR_VOT.m while the CPU version code is ./vot2017_trax/CFWCR_VOT_CPU.m, which the toolkit can call to get the results.
Before using the CFWCR, you need to fill the cuda(>=7.5) path in the install.m and run install.m to install all the dependences. 
It is found that the results can be different with different gpu models. Besides, the results change if using cudnn or without gpu. We test our tracker with GTX1080 GPU without cudnn. There are some bugs when using gpuDevice(1) in pascal gpu, which takes more than 100 seconds to execuate. The problem will be solved by setting "export CUDA_CACHE_MAXSIZE=8000000000" in the .~/bash_profile and restart the computer, or our gpu version code will be slower than that of our results.
Dependencies
1、matconvnet: 
The matconvnet(https://github.com/vlfeat/matconvnet.git) should be compile with cuda8.0. 
