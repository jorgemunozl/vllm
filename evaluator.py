import os


a = os.listdir("NFT/")

for i in range(65):
    with open(f"NFT/{a[i]}", "r") as f:
        content = f.read()
    with open(f"NFT/{i}.md", "w") as f:
        f.write(content)
    os.remove(f"NFT/{a[i]}")
