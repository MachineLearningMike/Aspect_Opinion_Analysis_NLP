import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

class VisualClass():
    def __init__(self):
        pass

    def QuickPlot(self, list, title = ''):
        plt.plot(list)
        plt.title(title)
        plt.show()

        plt.close()

    def PlotStepHistory(self, title, seriesDict, metaParam, imageSize = (4, 3), path = ''):
        fig, ax = plt.subplots(figsize = imageSize)

        ax.set_title(title)
        ax.set_xlabel('step')

        for seriesName in seriesDict:
            ax.plot( seriesDict[ seriesName ], label = seriesName )
        ax.legend(loc='best') # add all to the lagend

        import matplotlib.lines as mlines

        handles, labels = ax.get_legend_handles_labels()      
        handles = handles + [ mlines.Line2D( [], [], label = key + ": " + str(metaParam[key]) ) for key in metaParam]       
        ax.legend(handles = handles, loc = 'best', ncol = 4, frameon = True, framealpha = 0.5)
        ax.grid('on')

        if path == '' :
            plt.show()
        else:
            # the default is 100dpi for savefig:
            fig.savefig(path, dpi = (200)) 

        plt.close()
