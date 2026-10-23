import pickle
import matplotlib.pyplot as plt
try:
    import plot_results as pr
except Exception:
    pr = None

with open("submission_val_input_8.pkl", "rb") as f:
    results = pickle.load(f)

first_traj = results[0]
coords = first_traj['coords']
lons = [c[0] for c in coords] # extract longitude
lats = [c[1] for c in coords] # extract latitude

plt.figure(figsize=(10, 6))
plt.plot(lons, lats, 'b-', linewidth=1, label='Predicted trajectory')
plt.scatter(lons[::50], lats[::50], c='red', s=10, label='Sampled points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectory recovery visualization (geometry only)')
if pr and getattr(pr, 'FONT_PROP', None):
    plt.legend(prop=pr.FONT_PROP)
else:
    plt.legend()
plt.axis('equal')
plt.show()