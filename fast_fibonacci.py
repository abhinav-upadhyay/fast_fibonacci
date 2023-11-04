import matplotlib.pyplot as plt
import numpy as np

M = np.array([[1,1], [1,0]], dtype=np.float128)
TAO_1 = np.float128((1 + np.sqrt(5)) / 2)
TAO_2 = np.float128((1 - np.sqrt(5)) / 2)
INV_SQRT_FIVE =  1 / np.sqrt(5)


def _comput_M_n(n: int) -> np.ndarray:
    return np.linalg.matrix_power(M, n)


def fib_fast(n: int) -> int:
    M_n = _comput_M_n(n)
    result = np.matmul(M_n, np.array([1, 0]))
    return result[1]

def fib_closed_form(n: int) -> int:
    return INV_SQRT_FIVE * (np.power(TAO_1, n) - np.power(TAO_2, n))


def fib(n: int) -> int:
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    fib_prev = 0
    fib_current = 1
    
    for _ in range(2, n+1):
        fib_next = fib_prev + fib_current
        fib_prev, fib_current = fib_current, fib_next
    
    return fib_current

if __name__ == '__main__':
    # collect timings for fib and fib_fast for n going from 2 to 5000 in increments of 10
    import time
    n_max = 5000
    n_inc = 50
    fib_timings = {}
    fib_fast_timings = {}
    fib_closed_form_timings = {}
    for n in range(2, n_max+1, n_inc):
        start = time.time()
        fib_res = fib(n)
        end = time.time()
        fib_timings[n] = end - start



    for n in range(2, n_max+1, n_inc):
        start = time.time()
        fib_fast_res = fib_fast(n) 
        end = time.time()
        fib_fast_timings[n] = end - start


    for n in range(2, n_max+1, n_inc):

        start = time.time()
        fib_closed_form_res = fib_closed_form(n) 
        end = time.time()
        fib_closed_form_timings[n] = end - start


    # generate code to plot timings using matplotlib
    # Extract data for plotting
    n_values = list(fib_timings.keys())
    fib_timings_values = list(fib_timings.values())
    fib_fast_timings_values = list(fib_fast_timings.values())
    fib_closed_form_timings_values = list(fib_closed_form_timings.values())

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, fib_timings_values, label='fib(n) Timings', marker='o', linestyle='-', color='blue')
    plt.plot(n_values, fib_fast_timings_values, label='fib_fast(n) Timings', marker='o', linestyle='-', color='green')
    plt.plot(n_values, fib_closed_form_timings_values, label='fib_closed_form(n) Timings', marker='o', linestyle='-', color='red')

    # Label the axes and add a legend
    plt.xlabel('n')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of fib(n), fib_fast(n), and fib_closed_form(n)')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig(r'fib_timings.png')

    

    


    