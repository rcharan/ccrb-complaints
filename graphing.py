import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()
sns.set()

def graph_dists(counts, x_max, xs, ys):

  bins = range(x_max)
  fig, ax = plt.subplots()
  sns.distplot(counts, norm_hist = True, kde = False, bins = bins, ax = ax, hist_kws = {'alpha' : 0.3, 'label' : 'Data'})
  ax.set_xlabel('Number of Complaints')
  ax.set_ylabel('Frequency')
  ax.set_title('Frequency of Complaints')

  ax.bar(xs, ys, color = 'red', alpha = 0.3, label = 'Fit')
  ax.legend();

  return fig, ax