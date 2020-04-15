python/runImplicit.py --factors 64 --useGPU True --lambda 0.01 --iterations 40 --k 20 --numThreads 6
sleep(10)
python/runImplicit.py --factors 64 --useGPU True --lambda 0.03 --iterations 40 --k 20 --numThreads 6
sleep(10)
python/runImplicit.py --factors 64 --useGPU True --lambda 0.09 --iterations 40 --k 20 --numThreads 6
sleep(10)
python/runImplicit.py --factors 64 --useGPU True --lambda 0.27 --iterations 40 --k 20 --numThreads 6
