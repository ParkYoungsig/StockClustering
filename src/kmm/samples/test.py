import kmeans_clustering

print("loaded:", kmeans_clustering.__file__)

kms = kmeans_clustering.Kmms()
print("Kmms created, running...")

result = kms.run()

print("run() finished.")
print("roll_dir =", result["out"]["roll_dir"])
# ============================================================
