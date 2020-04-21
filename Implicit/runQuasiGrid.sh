CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 64 --useGPU True --lambda 0.01 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 64 --useGPU True --lambda 0.03 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 64 --useGPU True --lambda 0.09 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 64 --useGPU True --lambda 0.27 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 128 --useGPU True --lambda 0.01 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 128 --useGPU True --lambda 0.03 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 128 --useGPU True --lambda 0.09 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 128 --useGPU True --lambda 0.27 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 128 --useGPU True --lambda 0.81 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 128 --useGPU True --lambda 2.43 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 128 --useGPU True --lambda 7.29 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 128 --useGPU True --lambda 21.87 --iterations 40 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 64 --useGPU True --lambda 0.01 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 64 --useGPU True --lambda 0.03 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 64 --useGPU True --lambda 0.09 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 64 --useGPU True --lambda 0.27 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 128 --useGPU True --lambda 0.01 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 128 --useGPU True --lambda 0.03 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 128 --useGPU True --lambda 0.09 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 128 --useGPU True --lambda 0.27 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 128 --useGPU True --lambda 0.81 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 128 --useGPU True --lambda 2.43 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=1 python/runImplicit.py --factors 128 --useGPU True --lambda 7.29 --iterations 15 --k 20 --numThreads 6 &

CUDA_VISIBLE_DEVICES=0 python/runImplicit.py --factors 128 --useGPU True --lambda 21.87 --iterations 15 --k 20 --numThreads 6 &
