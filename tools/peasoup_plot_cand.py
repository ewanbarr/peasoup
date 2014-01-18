import peasoup_tools as pt
import sys
import pylab as plt

def main(overview_file,cand_id):
    overview = pt.OverviewFile(overview_file)
    plotter = pt.CandidatePlotter(overview)
    plotter.plot_cand(cand_id)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1],int(sys.argv[2]))
