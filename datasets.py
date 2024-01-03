metal=[f"datas/metal/fold{i}.npz" for i in range(6)]
pallet=[f"datas/pallet/light_on_{i}.npz" for i in range(5)]
std=[f"datas/std/emb{i}.npz" for i in range(5)]

datasets={}
for i,m in enumerate(metal):
    datasets[f"metal{i}"]=m
for i,p in enumerate(pallet):
    datasets[f"pallet{i}"]=p
for i,s in enumerate(std):
    datasets[f"std{i}"]=s

datasets["market"]="datas/market/market.npz"


