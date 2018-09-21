"""File di run."""

from subprocess import call
import sys

if __name__ == '__main__':
    num_bodies = int(sys.argv[1])
    
    # if len(sys.argv) == 5:
#         num_bodies = int(sys.argv[1])
#         num_iters_for_mean = int(sys.argv[2])
#         n_blocks = int(sys.argv[3])
#         threads_per_block = int(sys.argv[4])
#     elif len(sys.argv) == 3:
#         num_bodies = int(sys.argv[1])
#         num_iters_for_mean = int(sys.argv[2])
#
#         threads_per_block =  128
#         n_threads = num_bodies
#         n_blocks = (num_bodies + threads_per_block - 1) / threads_per_block
#
#         if n_blocks > 1024:
#             print("n_blocks is {}, too high -> setting to 1024".format(n_blocks))
#             n_blocks = 1024
#
#     else:
#         raise Exception("""Need exactly these parameters:
#         - n_bodies
#         - n_iters_for_mean
#         - n_blocks
#         - threads_per_block
#         """)
    # if len(sys.argv) == 3:
#         num_bodies = int(sys.argv[1])
#         num_iters_for_mean = int(sys.argv[2])
#     elif len(sys.argv) == 2:
#         num_bodies = int(sys.argv[1])
#         num_iters_for_mean = 30
#     else:
#         num_bodies = 8192
#         num_iters_for_mean = 30

    # print("\n\n ------ PYTHON ------")
#     print("\nExecuting {} times with {} bodies".format(num_iters_for_mean,
#                                                        num_bodies))

    iters = 10
    h = 0.05
    max_blocks = 8
    threads_per_block =  128
    
    try:
        call([
            "../build/x86_64/linux/release/nbody_cuda_naive",
            str(num_bodies), # num_bodies
            str(1), # n_blocks
            str(1), # threads_per_block
            str(iters), # iters
            str(h) # h
        ])
    except KeyboardInterrupt:
        print("\n\nSTOPPING\n\n")
        sys.exit("\nKeyboard interrupt\n")
    
    # for n_blocks in range(1, max_blocks + 1):
    #     print("\n\nExecuting with {} blocks of {} threads each".format(
    #         n_blocks, threads_per_block
    #     ))
    #
    #     try:
    #         call([
    #             "../build/x86_64/linux/release/nbody_cuda_naive",
    #             str(num_bodies),
    #             str(n_blocks),
    #             str(threads_per_block),
    #             str(iters),
    #             str(h)
    #         ])
    #     except KeyboardInterrupt:
    #         print("\n\nSTOPPING\n\n")
    #         sys.exit("\nKeyboard interrupt\n")

    # for 30 times
    # for n_run in range(1, num_iters_for_mean + 1):
#         print("\nPYTHON: run number {}".format(n_run))
#         print("Executing with {} blocks, each with {} threads, on {} bodies, with {} iterations".format(
#             n_blocks, threads_per_block, num_bodies, iters))
#
#         try:
#             call([
#                 "../build/x86_64/linux/release/nbody_cuda_naive",
#                 str(num_bodies),
#                 str(n_blocks),
#                 str(threads_per_block),
#                 str(iters),
#                 str(h)
#             ])
#         except KeyboardInterrupt:
#             print("\n\nSTOPPING\n\n")
#             sys.exit("\nKeyboard interrupt\n")

        print("Finished")
        print("##### ----- #####\n\n")
