import imageio
import glob

filenames = glob.glob('/Users/seabrook/Documents/FDL/Kernel/' + '*.png')

with imageio.get_writer('/Users/seabrook/Documents/FDL/Kernel/kernel.gif', mode='I', duration=0.04) as writer:
    for i in range(len(filenames) // 4):
        image = imageio.imread(filenames[i])
        writer.append_data(image)