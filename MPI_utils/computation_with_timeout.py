import concurrent.futures
import time

class ComputationWithTimeout:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds

    def run(self, mission=None):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if mission:
                future = executor.submit(mission)
            else:
                future = executor.submit(self.long_computation)
            try:
                # 等待计算完成，设置超时时间
                future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutException("Calculation took too long!")

    def long_computation(self):
        # 模拟一个计算复杂度高的步骤
        time.sleep(5)
 
class TimeoutException(Exception):
    pass

# 示例：设置超时时间为3秒
# computation = ComputationWithTimeout(timeout_seconds=3)

# 定义一个复杂计算任务
# def external_long_computation():
#     模拟一个计算复杂度高的步骤
#     time.sleep(5)

# try:
#     传递外部计算任务
#     computation.run(mission=external_long_computation)
# except TimeoutException as e:
#     print(e)