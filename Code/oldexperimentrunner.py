import os

for i in range(15, 29):
    print(f"\nexperiment {i}")
    os.system(f"git checkout experiment{i}")
    os.system("python PredFutureSales_Main.py")