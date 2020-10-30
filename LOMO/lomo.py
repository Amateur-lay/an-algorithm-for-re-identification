def lomo(images, *args):
    '''
    :param  images:a set of n RGB color images. Size: [h, w, 3, n]
    :param  option:
            optional parameters.
            numScales: number of pyramid scales in feature extraction. Default: 3
            blockSize: size of the sub-window for histogram counting. Default: 10
            blockStep: sliding step for the sub-windows. Default: 5
            hsvBins: number of bins for HSV channels. Default: [8,8,8]
            tau: the tau parameter in SILTP. Default: 0.3
            R: the radius paramter in SILTP. Specify multiple values for multiscale SILTP. Default: [3, 5]
            numPoints: number of neiborhood points for SILTP encoding. Default: 4
            The above default parameters are good for 128x48 and 160x60 person
            images. You may need to adjust the numScales, blockSize, and R parameters
            for other smaller or higher resolutions.
    :return: descriptors: the extracted LOMO descriptors. Size: [d, n]
    '''
    numScales = 3
    blockSize = 10
    blockStep = 5

    hsvBins = [8, 8, 8]
    tau = 0.3
    R = [3, 5]
    numPoints = 4
    #if nargin >= 2:
