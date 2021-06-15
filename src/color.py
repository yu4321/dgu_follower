import cv2
import numpy as np
import time
from sklearn.cluster import KMeans

class colorSorter():
    def make_histogram(self, cluster):
        """
        Count the number of pixels in each cluster
        :param: KMeans cluster
        :return: numpy histogram
        """
        t =time.time()
        #print('start make histogram ',t)
        numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        #print('complete arange ', time.time() - t)
        hist, _ = np.histogram(cluster.labels_, bins=numLabels)
        #print('complete histogram ', time.time() - t)
        hist = hist.astype('float32')
        hist /= hist.sum()
        #print('all complete')
        return hist

    def make_bar(self, height, width, color):
        """
        Create an image of a given color
        :param: height of the image
        :param: width of the image
        :param: BGR pixel values of the color
        :return: tuple of bar, rgb values, and hsv values
        """
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])
        hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv_bar[0][0]
        return bar, (red, green, blue), (hue, sat, val)

    def sort_hsvs(self, hsv_list):
        """
        Sort the list of HSV values
        :param hsv_list: List of HSV tuples
        :return: List of indexes, sorted by hue, then saturation, then value
        """
        bars_with_indexes = []
        for index, hsv_val in enumerate(hsv_list):
            bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
        bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
        return [item[0] for item in bars_with_indexes]

    def dominant_color(self, img):
        t = time.time()
        #print('start dominant color', t)
        height, width, _ = np.shape(img)
       # print('img height width ',height,width)
        # reshape the image to be a simple list of RGB pixels
        image = img.reshape((height * width, 3))
        #print('reshape complete ', time.time() - t)
        # we'll pick the 1 most common colors
        num_clusters = 1
        clusters = KMeans(n_clusters=num_clusters)
        #print('kmeans complete ', time.time() - t)
        clusters.fit(image)

        # count the dominant colors and put them in "buckets"
        histogram = self.make_histogram(clusters)
        #print('make histogram complete ', time.time() - t)
        # then sort them, most-common first
        combined = zip(histogram, clusters.cluster_centers_)
        #print('zip complete ', time.time() - t )
        combined = sorted(combined, key=lambda x: x[0], reverse=True)
        #print('sorted complete ', time.time() - t)
        # finally, we'll output a graphic showing the colors in order
        bars = []
        hsv_values = []
        for index, rows in enumerate(combined):
            bar, rgb, hsv = self.make_bar(100, 100, rows[1])
            #print(f'Bar {index + 1}')
            #print(f'  RGB values: {rgb}')
            #print(f'  HSV values: {hsv}')
            hsv_values.append(hsv)
            bars.append(bar)
        #print('enumarate   complete ', time.time() - t)
        # sort the bars[] list so that we can show the colored boxes sorted
        # by their HSV values -- sort by hue, then saturation
        sorted_bar_indexes = self.sort_hsvs(hsv_values)
        sorted_bars = [bars[idx] for idx in sorted_bar_indexes]
        #print('hsvs complete', time.time() - t)
        #cv2.imshow('Sorted by HSV values', np.hstack(sorted_bars))
        #cv2.imshow(f'{num_clusters} Most Common Colors', np.hstack(bars))
        #cv2.waitKey(0)
        return hsv[0]

    #이미지 4분할 후 h 채널 추출
    def img_crop(self, img):

        t =time.time()
        #print('start resize image ',t)
        img = cv2.resize(img, dsize=(40,60), interpolation = cv2.INTER_AREA)
        #print('end resize image ',time.time() - t)
        height, width, _ = np.shape(img)

        crop1 = img[0:int(height / 4), 0:int(width)]
        crop2 = img[int(height / 4):int(height / 2), 0:int(width)]
        crop3 = img[int(height / 2):int(height / 4 * 3), 0:int(width)]
        crop4 = img[int(height * 3 / 4):height, 0:int(width)]

        h1 = self.dominant_color(crop1)
        h2 = self.dominant_color(crop2)
        h3 = self.dominant_color(crop3)
        h4 = self.dominant_color(crop4)

        #print(h1, h2, h3, h4)

        return (h1, h2, h3, h4)

    def img_crop_and_get_dominants(self, img):
        return True

if __name__ == '__main__':
    color = colorSorter()
    img =  cv2.imread('test.jpg')
    t = time.time()
    print('start read at ',t)
    res = color.img_crop(img)
    print('res : ',res,', elapsed : ',time.time() - t)