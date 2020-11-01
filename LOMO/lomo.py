'''
lomo函数主要是用于将一组图像数据集images的特征提取出来
值得注意的一点是opencv的储存是brg，matlab中的储存是rgb
'''
import numpy as np
import cv2


# import retinex


class lomo():
    def __init__(self, images, numScales=3, blockSize=10, blockStep=5, hsvBins=None, tau=0.3, R=None, numPoints=4):
        if R is None:
            R = np.array([3, 5])
        if hsvBins is None:
            hsvBins = np.array([8, 8, 8])
        self.images = images
        self.numScales = numScales
        self.blockSize = blockSize
        self.blockStep = blockStep
        self.hsvBins = hsvBins
        self.tau = tau
        self.R = R
        self.numPoints = numPoints

    def lomo(self):
        fea1 = self.PyramidMaxJointHist(self.images, self.numScales, self.blockSize, self.blockStep, self.hsvBins)
        fea2 = []
        for i in range(len(self.R)):
            fea2 = [[fea2], [
                self.PyramidMaxSILTPHist(self.images, self.numScales, self.blockSize, self.blockStep, self.tau,
                                         self.R[i], self.numPoints)]]
        descriptors = [[fea1], [fea2]]
        return descriptors

    def PyramidMaxJointHist(self, ori_imgs, numScales=3, blockSize=10, blockStep=5, colorBins=None):
        if colorBins is None:
            colorBins = np.array([8, 8, 8])
        totalBins = np.prod(colorBins)
        numImgs = ori_imgs.shape[3]
        images = np.zeros(ori_imgs.shape)
        for i in range(numImgs):
            I = ori_imgs[:, :, :, i]
            # I = retinex(I)
            #I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
            I[:, :, 0] = np.minimum(np.floor(I[:, :, 0] * colorBins[0]), colorBins[0] - 1)
            I[:, :, 1] = np.minimum(np.floor(I[:, :, 1] * colorBins[1]), colorBins[1] - 1)
            I[:, :, 2] = np.minimum(np.floor(I[:, :, 2] * colorBins[2]), colorBins[2] - 1)
            images[:, :, :, i] = I
        minRow = 0
        minCol = 0
        descriptors = []
        for i in range(numScales):
            patterns = images[:, :, 2, :] * colorBins[1] * colorBins[0] + images[:, :, 1, :] * colorBins[
                0] + images[:, :, 0, :]
            patterns = np.reshape(patterns, (int(np.size(patterns) / numImgs),numImgs))
            height = np.size(patterns, 0)
            width = np.size(patterns, 1)
            maxRow = height - blockSize + 1
            maxCol = width - blockSize + 1
            [cols, rows] = np.meshgrid(np.arange(minCol, maxCol + 1, blockStep),
                                       np.arange(minRow, maxRow + 1, blockStep))
            cols = cols[:]
            rows = rows[:]
            numBlocks = len(cols)
            numBlocksCol = len(np.arange(minCol, maxCol + 1, blockStep))
            if numBlocks == 0:
                break
            offset = np.transpose(np.arange(0, blockSize)) + np.dot(np.arange(0, blockSize), height)
            index = np.ravel_multi_index((rows, cols), (height, width), order="F")
            index = np.transpose(offset) + index
            patches = patterns[index[:], :]
            patches = np.reshape(patches, (np.size(patches) / (numBlocks * numImgs), numBlocks * numImgs))
            fea = np.histogram(patches, np.arange(0, totalBins))
            fea = np.reshape(fea, (totalBins, numBlocks / numBlocksCol, numBlocksCol, numImgs))
            fea = np.max(fea, axis=2)
            fea = np.reshape(fea, (np.size(fea) / numImgs, numImgs))
            descriptors = [[descriptors], [fea]]
            if i < numScales:
                images = self.ColorPooling(images, 'average')
        descriptors = np.log(descriptors + 1)
        descriptors = np.normc(descriptors)
        return descriptors

    def ColorPooling(self, images, method):
        height, width, numChannels, numImgs = images.shape
        outImages = images
        if height % 2 == 1:
            outImages[-1, :, :, :] = []
            height = height - 1
        if width % 2 == 1:
            outImages[:, -1, :, :] = []
            width = width - 1
        if height == 0 and width == 0:
            raise IndexError('error:Over scaled image: height=%d, width=%d.', height, width)
        height = height / 2
        width = width / 2
        outImages = np.reshape(outImages, (2, height, 2, width, numChannels, numImgs))
        outImages = np.swapaxes(outImages, 0, 4)
        outImages = np.swapaxes(outImages, 0, 2)
        outImages = np.swapaxes(outImages, 0, 1)
        outImages = np.swapaxes(outImages, 1, 3)
        outImages = np.swapaxes(outImages, 3, 5)
        outImages = np.reshape(outImages, (height, width, numChannels, numImgs, 2 * 2))
        if method == "average":
            outImages = np.mean(outImages, 3)
        elif method == "max":
            outImages = np.max(outImages, axis=3)
        else:
            raise KeyError('Error pooling method: %s.', method)
        return outImages

    def PyramidMaxSILTPHist(self, oriImgs, numScales=3, blockSize=10, blockStep=5, tau=0.3, R=5, numPoints=4):
        totalBins = 3 ** numPoints
        imgHeight, imgWidth, numImgs = oriImgs.shape[0], oriImgs.shape[1], oriImgs.shape[3]
        images = np.zeros(imgHeight, imgWidth, numImgs)
        for i in range(numImgs):
            I = oriImgs[:, :, :, i]
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            images[:, :, i] = np.float64(I) / 255
        minRow = 0
        minCol = 0
        descriptors = []
        for i in range(numScales):
            height = images.shape[0]
            width = images.shape[1]
            if width < R * 2 + 1:
                print('Skip scale R = %d, width = %d.\n', R, width)
                continue
            patterns = self.SILIP(images, tau, R, numPoints)
            patterns = np.reshape(patterns, (int(np.size(patterns) / numImgs), numImgs))
            maxRow = height - blockSize + 1
            maxCol = width - blockSize + 1
            [cols, rows] = np.meshgrid(np.arange(minCol, maxCol + 1, blockStep),
                                       np.arange(minRow, maxRow + 1, blockStep))
            cols = cols[:]
            rows = rows[:]
            numBlocks = len(cols)
            numBlocksCol = len(np.arange(minCol, maxCol + 1, blockStep))
            if numBlocks == 0:
                break
            offset = np.transpose(np.arange(0, blockSize)) + np.dot(np.arange(0, blockSize), height)
            index = np.ravel_multi_index((rows, cols), (height, width), order="F")
            index = np.transpose(offset) + index
            patches = patterns[index[:], :]
            patches = np.reshape(patches, (np.size(patches) / (numBlocks * numImgs), numBlocks * numImgs))
            fea = np.histogram(patches, np.arange(0, totalBins))
            fea = np.reshape(fea, (totalBins, numBlocks / numBlocksCol, numBlocksCol, numImgs))
            fea = np.max(fea, axis=2)
            fea = np.reshape(fea, (np.size(fea) / numImgs, numImgs))
            descriptors = [[descriptors], [fea]]
            if i < numScales:
                images = self.ColorPooling(images, 'average')
        descriptors = np.log(descriptors + 1)
        descriptors = np.normc(descriptors)
        return descriptors

    def SILIP(self, I, tau=0.03, R=0, numPoints=4, encoder=0):
        if tau <= 0 or np.floor(R) != R or R < 0 or not (numPoints == 4 or numPoints == 8) or not (
                encoder == 0 or encoder == 1):
            raise KeyError('Error parameter values!')
        h, w, n = I.shape
        if h < 2 * R + 1 or w < 2 * R + 1:
            raise KeyError('Too small image or too large R!')
        I0 = np.zeros(h + 2 * R, w + 2 * R, n)
        I0[R + 1: -1 - R + 1, R + 1: -1 - R + 1, :] = np.float64(I)
        I0[0: R + 1, :, :] = np.repmat(I0[R + 1, :, :], [R, 1, 1])
        I0[-1 - R + 1:, :, :] = np.repmat(I0[-1 - R, :, :], [R, 1, 1])
        I0[:, 1: R + 1, :] = np.repmat(I0[:, R + 1, :], [1, R, 1])
        I0[:, -1 - R + 1:, :] = np.repmat(I0[:, -1 - R, :], [1, R, 1])
        I1 = I0[R + 1:-1 - R + 1, 2 * R + 1:, :]
        I3 = I0[0:- 2 * R, R + 1: -1 - R + 1, :]
        I5 = I0[R + 1:-1 - R + 1, 0: - 2 * R, :]
        I7 = I0[2 * R + 1:, R + 1: -1 - R + 1, :]
        if numPoints == 8:
            I2 = I0[0:-1 - 2 * R, 2 * R + 1:, :]
            I4 = I0[0:-1 - 2 * R, 0: -1 - 2 * R + 1, :]
            I6 = I0[2 * R + 1:, 0: -1 - 2 * R + 1, :]
            I8 = I0[2 * R + 1:, 2 * R + 1:, :]
        L = (1 - tau) * I
        U = (1 + tau) * I
        if encoder == 0:
            if numPoints == 4:
                J = (I1 < L).astype(int) + (I1 > U).astype(int) * 2 + (
                            (I3 < L).astype(int) + (I3 > U).astype(int) * 2) * 3 + (
                                (I5 < L).astype(int) + (I5 > U).astype(int) * 2) * 9 + (
                            (I7 < L).astype(int) + (I7 > U).astype(int) * 2) * 27
            else:
                J = (I1 < L).astype(int) + (I1 > U).astype(int) * 2 + (
                            (I2 < L).astype(int) + (I2 > U).astype(int) * 2) * 3 + (
                                (I3 < L).astype(int) + (I3 > U).astype(int) * 2) * 3 ** 2 + (
                            (I4 < L).astype(int) + (I4 > U).astype(int) * 2) * 3 ** 3 + (
                                (I5 < L).astype(int) + (I5 > U).astype(int) * 2) * 3 ** 4 + (
                                (I6 < L).astype(int) + (I6 > U).astype(int) * 2) * 3 ** 5 + (
                            (I7 < L).astype(int) + (I7 > U).astype(int) * 2) * 3 ** 6 + (
                                (I8 < L).astype(int) + (I8 > U).astype(int) * 2) * 3 ** 7

        else:
            if numPoints == 4:
                J = (I1 > U) + (I1 < L) * 2 + (I3 > U) * 2 ** 2 + (I3 < L) * 2 ** 3 + (I5 > U) * 2 ** 4 + (
                            I5 < L) * 2 ** 5 + (
                            I7 > U) * 2 ** 6 + (I7 < L) * 2 ** 7
            else:
                J = (I1 > U) + (I1 < L) * 2 + (I2 > U) * 2 ** 2 + (I2 < L) * 2 ** 3 + (I3 > U) * 2 ** 4 + (
                            I3 < L) * 2 ** 5 + (
                            I4 > U) * 2 ** 6 + (I4 < L) * 2 ** 7 + (I5 > U) * 2 ** 8 + (I5 < L) * 2 ** 9 + (
                                I6 > U) * 2 ** 10 + (I6 < L) * 2 ** 11 + (I7 > U) * 2 ** 12 + (
                            I7 < L) * 2 ** 13 + (I8 > U) * 2 ** 14 + (I8 < L) * 2 ** 15
