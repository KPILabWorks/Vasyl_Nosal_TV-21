import time

class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        print(f"Час виконання: {self.elapsed_time:.6f} сек.")

with Timer():
    total = sum(range(100 ** 4))
    print("total =", total)
