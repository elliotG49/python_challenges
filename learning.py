#
# example1.py
#

import threading
import time
import multiprocessing


def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def find_primes(start, end):
    """Find all prime numbers in the given range."""
    primes = []
    for num in range(start, end + 1):
        if is_prime(num):
            primes.append(num)
    return primes


def worker(worker_id, start, end):
    """Worker function to find primes in a specific range."""
    print(f"Worker {worker_id} starting")
    primes = find_primes(start, end)
    print(f"Worker {worker_id} found {len(primes)} primes")


def main():
    """Main function to coordinate the multi-threaded prime search."""
    start_time = time.time()

    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    # Define the range for prime search
    total_range = 2_000_000
    chunk_size = total_range // num_cores

    threads = []
    # Create and start threads equal to the number of cores
    for i in range(num_cores):
        start = i * chunk_size + 1
        end = (i + 1) * chunk_size if i < num_cores - 1 else total_range
        thread = threading.Thread(target=worker, args=(i, start, end))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Calculate and print the total execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"All workers completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
