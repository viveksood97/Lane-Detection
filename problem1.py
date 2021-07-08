import cv2
import numpy as np
import time
import matplotlib.pyplot as plt



class ImageOperations:

    def __init__(self,frame):
        self.frame = frame
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def deNoiser(self):
        ret,thresh4 = cv2.threshold(self.frame,8,255,cv2.THRESH_TOZERO)
        
        thresh4[thresh4 > 180] = 255
        return thresh4

    def applyHistogramEqualization(self):
        img = self.deNoiser()
        #imageLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        b,g,r = cv2.split(img)
    
        b_new = self.cdf_histogram(b.flatten())
        g_new = self.cdf_histogram(g.flatten())
        r_new = self.cdf_histogram(r.flatten())

        
        result = cv2.merge((b_new,g_new,r_new))

        #imageBGR = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)  
        
        
        return result

    def cdf_histogram(self,arr):
        histogram = np.zeros(256)
        histogram_with_cdf = np.zeros(256)
        for pixel in arr:
            histogram[pixel] += 1

        
        histogram_with_cdf[0] = histogram[0]

        for i in range(1,256):
            histogram_with_cdf[i] = histogram_with_cdf[i-1] + histogram[i]

        
        minIntensity = min(histogram_with_cdf)
        maxIntensity = max(histogram_with_cdf)
        
        for i,val in enumerate(histogram_with_cdf):
            histogram_with_cdf[i] = (255*(val - minIntensity))/(maxIntensity-minIntensity)

        histogram_with_cdf=histogram_with_cdf.astype('uint8')
        img_new = histogram_with_cdf[arr]

        img1 = np.reshape(img_new, self.gray.shape)
        blur = cv2.GaussianBlur(img1, (5, 5), 0)
        median = cv2.medianBlur(blur, 5)

        return median





def main():
    start = time.time()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    cap = cv2.VideoCapture('./input/Night Drive - 2689.mp4')


    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter('./output.avi', codec, 20, (1920,1080))

    count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        else:
            image = ImageOperations(frame)
            result = image.applyHistogramEqualization()
            
            
            vis = np.concatenate((frame, result), axis=1)
            scale_percent = 50 # percent of original size
            width = int(vis.shape[1] * scale_percent / 100)
            height = int(vis.shape[0] * (scale_percent*2) / 100)
            
            dim = (width, height)
            # resize image
            resized = cv2.resize(vis, dim, interpolation = cv2.INTER_AREA)
            cv2.putText(resized,'Original Video', (40,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(resized,'Output', (1100,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            count += 1
            #cv2.imwrite("frame%d.jpg" % count, resized)
            out.write(resized)
            #cv2.imshow("frame",resized)
            


            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    end = time.time()
    print(end-start)


if __name__ == '__main__':
    main()
