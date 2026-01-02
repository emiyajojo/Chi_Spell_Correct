import os
import sys
import time

start_time = time.time()
os.system('curl localhost:4050/correction -X POST -d \'{"query":"安佘尔没有主力资金为神么股价上张"}\' --header "Content-Type:application/json"')
end_time = time.time()
print("\n\n单条样本耗时:", end_time - start_time)