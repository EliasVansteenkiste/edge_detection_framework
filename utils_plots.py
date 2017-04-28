import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print plt.get_backend() 
import warnings
import numpy as np
import matplotlib.animation as animation
import utils

warnings.simplefilter('ignore')
anim_running = True

def show_img(img, img_dir='test.jpg'):
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    a = a.set_title(img_dir)
    plt.imshow(img)
    fig.savefig(img_dir, bbox_inches='tight')


def plot_learning_curves(train_losses, valid_losses, expid, img_dir):
    fig = plt.figure()
    x_range = np.arange(len(train_losses)) + 1

    plt.plot(x_range, train_losses)
    plt.plot(x_range, valid_losses)

    if img_dir is not None:
        fig.savefig(img_dir + '/%s.png' % expid, bbox_inches='tight')
        print 'Saved plot'
    else:
        plt.show()
    fig.clf()
    plt.close('all')


def plot_img(img, outfile):
    '''
    Plot the three color bands for an image

    Automatically detect the number of color bands the 
    image has, and plot the image so that it is easy to 
    visualize all the color bands
    '''

    # print fileName

    # img      = io.imread(fileName)
    numBands = img.shape[0]
    print img.shape

    plt.figure(figsize=(5*(numBands+1), 5)) # One for the original image
    f = 1.0/(numBands+1)
    for b in range(numBands):
        plt.axes( [b * f, 0, f, 1] )
        plt.imshow(img[ b, :, :], cmap=plt.cm.viridis)
        plt.xticks([]); plt.yticks([])

    if numBands == 4:
        img1 = img[ :-1, :, :].copy()
        img1 = img1*255.0/img1.max()
        img2 = np.zeros(np.shape(img1))
        img2[0,:,:] = img1[2,:,:]
        img2[1,:,:] = img1[1,:,:]
        img2[2,:,:] = img1[0,:,:]
        img2 = img2.astype(np.uint8)
    else:
        img2 = img

    plt.axes( [(b+1) * f, 0, f, 1] )
    plt.imshow(np.swapaxes(img2,0,2))
    plt.xticks([]); plt.yticks([])

    plt.savefig(outfile)
    plt.close('all')

    return img
    
