"""File di run."""

from subprocess import call
import sys

if __name__ == '__main__':
    if len(sys.argv) == 5:
        num_bodies = int(sys.argv[1])
        num_iters_for_mean = int(sys.argv[2])
        n_blocks = int(sys.argv[3])
        threads_per_block = int(sys.argv[4])
    elif len(sys.argv) == 3:
        num_bodies = int(sys.argv[1])
        num_iters_for_mean = int(sys.argv[2])
        
        threads_per_block =  256
        n_threads = num_bodies
        n_blocks = (n_threads + threads_per_block - 1) / threads_per_block
    
        if n_blocks > 1024:
            print("n_blocks is {}, too high -> setting to 1024".format(n_blocks))
            n_blocks = 1024

    else:
        raise Exception("""Need exactly these parameters:
        - n_bodies
        - n_iters_for_mean
        - n_blocks
        - threads_per_block
        """)
    # if len(sys.argv) == 3:
#         num_bodies = int(sys.argv[1])
#         num_iters_for_mean = int(sys.argv[2])
#     elif len(sys.argv) == 2:
#         num_bodies = int(sys.argv[1])
#         num_iters_for_mean = 30
#     else:
#         num_bodies = 8192
#         num_iters_for_mean = 30

    print("\n\n ------ PYTHON ------")
    print("\nExecuting {} times with {} bodies".format(num_iters_for_mean,
                                                       num_bodies))

    iters = 1000
    h = 0.05

    # for 30 times
    for n_run in range(1, num_iters_for_mean + 1):
        print("\nPYTHON: run number {}".format(n_run))
        print("Executing with {} blocks, each with {} threads, on {} bodies, with {} iterations".format(
            n_blocks, threads_per_block, num_bodies, iters))

        try:
            call([
                "../build/x86_64/linux/release/nbody_cuda_naive",
                str(num_bodies),
                str(n_blocks),
                str(threads_per_block),
                str(iters),
                str(h)
            ])
        except KeyboardInterrupt:
            print("\n\nSTOPPING\n\n")
            sys.exit("\nKeyboard interrupt\n")

        print("Finished")
        print("##### ----- #####\n\n")
