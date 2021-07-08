import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import random
import os


# Drawing Functions
# self.debug_frame = cv2.line(self.debug_frame, (0, height-(i+1)*ratio), (1280, height-(i+1)*ratio), (0, 255, 0), thickness=2)
# self.debug_frame = cv2.rectangle(self.debug_frame,(max_val-threshold,self.height-(index+1)*self.ratio),(max_val+threshold,self.height-(index)*self.ratio),(0,255,0),3)
# for j in range(len(closestX)):
#   self.debug_frame = cv2.circle(self.debug_frame, (int(closestX[j]),int(closestY[j])), radius=0, color=(0, 0, 255), thickness=10)

class Debugger:

    def __init__(self,size):
        self.stack = []
        self.type = []
        self.nrows = size[0]
        self.ncols = size[1]
    
    def collect(self,frame,frame_type):
        self.stack.append(frame)
        self.type.append(frame_type)

    def display(self,plot=False):
        if plot == True:
            for index in range(0,len(self.stack)):
                if self.type[index][0] == "image":
                    ax = plt.subplot(self.nrows,self.ncols,index+1)
                    ax.imshow(cv2.cvtColor(self.stack[index], cv2.COLOR_BGR2RGB), cmap='gray')
                elif self.type[index][0] == "plot":
                    ax = plt.subplot(self.nrows,self.ncols,index+1)
                    ax.plot(*zip(*self.stack[index]))
                elif self.type[index][0] == "scatter":
                    ax = plt.subplot(self.nrows,self.ncols,index+1)
                    ax.scatter(*zip(*self.stack[index]),linewidths=0.1)

                
            plt.tight_layout()
            plt.show()
        else:
            stack = list()
            temp = list()
            for index,image in enumerate(self.stack):
                if(self.type[index][1] == "binary"):
                    temp.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
                else:
                    temp.append(image)
                if len(temp) == 3:
                    stack.append(np.hstack((temp[0],temp[1],temp[2])))
                    temp = list()
                
            stacked = np.vstack(tuple(stack))
            cv2.imshow('Debugger',cv2.resize(stacked,None,fx=0.4,fy=0.4))
            
class PreviousState:

    def __init__(self):
        self.state = {"prev_max_right_x":0,
                      "prev_state_closest_right_X": [],
                      "prev_state_closest_right_Y": [],
                      "previous_model_right": [],
                      "previous_model_left": [],
                      "previous_dist_variation_top": [],
                      "previous_dist_variation_bottom": [],
                      "avg_previous_variation" : 0,
                      "previous_coef_left":0,
                      "previous_coef_right":0}

    def update(self,key,value):
        self.state[key] = value

    def add_prev_dist_variation(self,val_top,val_bottom):
        self.state["previous_dist_variation_top"].append(val_top)
        self.state["previous_dist_variation_bottom"].append(val_bottom)


    def update_prev_dist_variation(self):
        self.state["avg_previous_variation"] = abs(np.mean(self.state["previous_dist_variation_top"]) - np.mean(self.state["previous_dist_variation_bottom"]))
    

class ImageOperations:

    def __init__(self,frame):
        self.frame = frame
        #Camera Parameters
        self.K =  np.asarray([[ 9.037596e+02, 0.00000000e+00, 6.957519e+02],
                            [0.00000000e+00, 9.019653e+02, 2.242509e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.dist = np.asarray([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])

        #Homography
        # self.src = np.asarray([[309, 717], [584, 493], [745, 493], [1104, 717]], np.float32)
        # self.dst = np.array([[309, 717],[309, 0], [1104, 0], [1104, 717]], np.float32)

        self.src = np.array([[100, 512], [480, 300], [720, 300], [720, 512]], np.float32)
    

        self.dst = np.array([[300, 512], [300, 0], [980, 0], [980, 512]], np.float32)


        self.H = cv2.getPerspectiveTransform(self.src, self.dst)
        self.H_inv = cv2.getPerspectiveTransform(self.dst, self.src)


        self.unsharpening_kernel = -(1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                                          [4, 16, 24, 16, 4],
                                                          [6, 24, -476, 24, 6],
                                                          [4, 16, 24, 16, 4],
                                                          [1, 4, 6, 4, 1]])

        self.sharpening_kernel = np.array([[-3,-3,-3,-3,-3],[-3,-2,1,-2,-3],[-3,-1,58,-1,-3],[-3,-2,1,-2,-3],[-3,-3,-3,-3,-3]])

    def undistort(self):
        undistorted_img = cv2.undistort(self.frame, self.K, self.dist, None, self.K)
        return undistorted_img

    def clahe(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe_object = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
        image[:,:,0] = clahe_object.apply(image[:,:,0])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        return image

    def sharpen(self,image,kernel):
        image = cv2.filter2D(image, -1, kernel)
        return image

    def color_filter(self,image):
        image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
        lower_yellow = np.array([0, 0, 0], dtype="uint8")
        upper_yellow = np.array([0, 0, 0], dtype="uint8")

        lower_white = np.array([200,200,200], dtype=np.uint8)
        upper_white = np.array([255,255,255], dtype=np.uint8)

        mask1 = cv2.inRange(image_HSV, lower_yellow, upper_yellow)
        mask2 = cv2.inRange(image, lower_white, upper_white)

        return mask1,mask2

    def warpPerspective(self,image,H):
        unwarped_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return unwarped_image


class Line_fit:

    def __init__(self,frame, row_bunches, prev_max_right_x, prev_state_closest_right_X, prev_state_closest_right_Y,debug_frame=False):
        self.frame = frame
        self.row_bunches = row_bunches
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.ratio = round(self.height/row_bunches)
        self.prev_max_right_x = prev_max_right_x
        self.prev_state_closest_right_X = prev_state_closest_right_X
        self.prev_state_closest_right_Y = prev_state_closest_right_Y
        self.debug_frame = debug_frame


    def find_max_x(self,frame_snippet):

        global_max_count = np.count_nonzero(self.frame == 255, axis=0)
        global_lanes_point_split = np.split(global_max_count, 2)
        global_left_max_x = np.argmax(global_lanes_point_split[0])
        global_right_max_x = np.argmax(global_lanes_point_split[1]) 

        lane_point_count = np.count_nonzero(frame_snippet == 255, axis=0)
        lanes_point_split = np.split(lane_point_count, 2)
        left_max_x = np.argmax(lanes_point_split[0])
        right_max_x = np.argmax(lanes_point_split[1])

        if(abs(global_right_max_x - right_max_x)>160):
            right_max_x = global_right_max_x

        if(abs(global_left_max_x - left_max_x)>160):
            left_max_x = global_left_max_x

        if(lanes_point_split[1][right_max_x] > 40):
            max_right = right_max_x + len(lanes_point_split[0])
            
            self.prev_max_right_x = max_right
        else:
            max_right = self.prev_max_right_x

        #print(lanes_point_split[1][right_max_x],right_max_x + len(lanes_point_split[0]),max_right)

        return left_max_x, right_max_x + len(lanes_point_split[0])

    def calculate_weights(self,points):
        length = len(points)
        ratio = 1/length
        total_sum = 0
        weights = []

        def split_list(alist, wanted_parts):
            length = len(alist)
            return [alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts)]

        if(length == 1):
            weights.append(1)
            return weights
        parts = split_list(points, 2)
        
        for i in range(len(parts[0])):
            total_sum += ratio
            weights.append(total_sum)
        for i in range(len(parts[1])):
            total_sum -= ratio
            if (total_sum > 0):
                weights.append(total_sum)
            else:
                weights.append(0)
        
        return weights


    def good_points(self,max_val,frame_snippet,index):

        threshold = 100
        closestX = []
        closestY = []

        roi = frame_snippet[:,max_val-threshold:max_val+threshold]

        points_for_line = np.nonzero(roi)

        translateY = round(self.height-(index+1)*self.ratio)
        translateX = max_val-threshold

        if(len(points_for_line[0]) == 0):
            newx = "none"
            newy = "none"
            return newx, newy,  closestX, closestY

        weightsX = self.calculate_weights(points_for_line[1])
        weightsY = self.calculate_weights(points_for_line[0])
        
        y = round(np.average(points_for_line[0], weights=weightsX))
        x = round(np.average(points_for_line[1], weights=weightsY))

        newy = y + translateY
        newx = x + max_val-threshold
        
        
        return newx, newy, points_for_line[1] + max_val-threshold, points_for_line[0] + translateY

    @staticmethod
    def radius_of_curvature(coeficents, y):
        coef1, coef2, _ = coeficents

        roc = ((1 + (2*coef1*y + coef2)**2) **(3/2))/ (2*coef1)

        roc = np.min(roc)
        return roc

    
    def findPoints(self):

        leftx = []
        lefty = []
        rightx = []
        righty = []

        frame = self.frame
        height = self.height
        ratio = self.ratio


        left_max_x, right_max_x = self.find_max_x(frame[height-(0+1)*315:height-(0),:])


        for i in range(self.row_bunches):

            frame_snippet = frame[height-(i+1)*ratio:(height-(i)*ratio),:]

            
            #self.debug_frame = cv2.line(self.debug_frame, (right_max_x, 0), (right_max_x, height), (0, 255, 0), thickness=2)
            
            
            if(i == 0):
                
                new_left_x, new_left_y, closest_left_X, closest_left_Y = self.good_points(left_max_x,frame_snippet,i)
                
                if new_left_x == "none":
                    new_left_x = left_max_x
                else:
                    left_max_x = int((new_left_x + left_max_x)/2)
                
                new_right_x, newy_right_y, closest_right_X,closest_right_Y = self.good_points(right_max_x,frame_snippet,i)
                prev_state_closest_right_X = closest_right_X
                prev_state_closest_right_Y = closest_right_Y
                # if(self.flag == False):
                #     self.prev_state_closest_right_X = closest_right_X
                #     self.prev_state_closest_right_Y = closest_right_Y
                # else:
                #     closest_right_X = self.prev_state_closest_right_X
                #     closest_right_Y = self.prev_state_closest_right_Y 

                if new_right_x == "none":
                    new_right_x = right_max_x
                else:
                    right_max_x = int((new_right_x + right_max_x)/2)
            else:
                new_left_x, new_left_y, closest_left_X, closest_left_Y  = self.good_points(left_max_x,frame_snippet,i)
                new_right_x, newy_right_y, closest_right_X,closest_right_Y = self.good_points(right_max_x,frame_snippet,i)

                if(new_left_x != "none"):
                    left_max_x = new_left_x
                    leftx.extend(closest_left_X)
                    lefty.extend(closest_left_Y)
                if(new_right_x != "none"):
                    right_max_x = new_right_x
                    rightx.extend(closest_right_X)
                    righty.extend(closest_right_Y)

            

            #self.debug_frame = cv2.line(self.debug_frame, (0, height-(i+1)*ratio), (1280, height-(i+1)*ratio), (255, 255, 255), thickness=2)

            self.debug_frame = cv2.rectangle(self.debug_frame,(left_max_x-50,self.height-(i+1)*self.ratio),(left_max_x+50,self.height-(i)*self.ratio),(0,255,0),3)
            self.debug_frame = cv2.rectangle(self.debug_frame,(right_max_x-50,self.height-(i+1)*self.ratio),(right_max_x+50,self.height-(i)*self.ratio),(0,255,0),3)
        

            for j in range(len(closest_left_X)):
                self.debug_frame = cv2.circle(self.debug_frame, (int(closest_left_X[j]),int(closest_left_Y[j])), radius=0, color=(0, 0, 255), thickness=2)
            for j in range(len(closest_right_X)):
                self.debug_frame = cv2.circle(self.debug_frame, (int(closest_right_X[j]),int(closest_right_Y[j])), radius=0, color=(0, 0, 255), thickness=2)

        return leftx, lefty, rightx, righty, self.debug_frame
        

        
        

def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    previous = PreviousState()

    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter('problem2part1.avi',codec, 20.0, (1392,512))
    images=[]

    for image in os.listdir("./input/data"):
        if image.endswith(".png"):
            images.append(os.path.join("./input/data", image))

    for image in images:
        frame = cv2.imread(image)
        debug = Debugger((1,1))

        #debug.collect(frame,("image","color"))
        image = ImageOperations(frame)
        
        undistorted_img = image.undistort()
        #debug.collect(undistorted_img,("image","color"))

        clahe_img = image.clahe(undistorted_img)
        #debug.collect(clahe_img,("image","color"))

        sharpened_image = image.sharpen(clahe_img,image.unsharpening_kernel)
        #debug.collect(sharpened_image,("image","color"))

        yellow_mask, white_mask = image.color_filter(sharpened_image)
        #debug.collect(yellow_mask,("image","binary"))
        #debug.collect(white_mask,("image","binary"))

        combined_mask = yellow_mask | white_mask
        #debug.collect(combined_mask,("image","binary"))

        binary_unwarp = image.warpPerspective(combined_mask,image.H)
        #debug.collect(binary_unwarp,("image","binary"))


        unwarped_image= image.warpPerspective(sharpened_image,image.H)
        #debug.collect(unwarped_image,("image","color"))

        prev_max_right_x = previous.state["prev_max_right_x"]
        prev_state_closest_right_X = previous.state["prev_state_closest_right_X"]
        prev_state_closest_right_Y = previous.state["prev_state_closest_right_Y"]

        
        debug_frame = copy.deepcopy(unwarped_image)
        
        line = Line_fit(cv2.medianBlur(binary_unwarp, 5),16, prev_max_right_x, prev_state_closest_right_X, prev_state_closest_right_Y,debug_frame=debug_frame)
        leftx, lefty, rightx, righty, debug_frame_updated = line.findPoints()

        #debug.collect(debug_frame_updated,("image","gray"))

        # previous.update("prev_max_right_x",line.prev_max_right_x)
        # previous.update("prev_state_closest_right_X",line.prev_state_closest_right_X)
        # previous.update("prev_state_closest_right_Y",line.prev_state_closest_right_Y)
        
        # test = copy.deepcopy(unwarped_image)
        polyfill = copy.deepcopy(unwarped_image)
        
        # for i in range(len(leftx)):
        #     test = cv2.circle(test, (int(leftx[i]),int(lefty[i])), radius=0, color=(0, 0, 255), thickness=1)
        # for i in range(len(rightx)):
        #     test = cv2.circle(test, (int(rightx[i]),int(righty[i])), radius=0, color=(0, 0, 255), thickness=1)

        #debug.collect(test,("image","gray"))

        try:   
            z1 = np.polyfit(lefty, leftx, 2)
            z2 = np.polyfit(righty, rightx, 2)
        except Exception as e:
            hola = 3
            #print(e)
            #print(leftx,lefty)
            

        

        f1 = np.poly1d(z1)
        f2 = np.poly1d(z2)

        # calculate new x's and y's
        y_new_left = np.linspace(0, 719, 720)
        x_new_left = f1(y_new_left)

        y_new_right = y_new_left
        x_new_right = f2(y_new_right)

        
        


        left_variance_high = False
        right_variance_high = False
        left_variance = np.var(x_new_left)
        right_variance = np.var(x_new_right)

        # lis = [x for x in range(0,720)]

        # randomlist_top = lis[:200]
        # randomlist_bottom = lis[200:]
        

        # average_dist_top = []
        # average_dist_bottom = []

        # for ix in randomlist_top:
        #     average_dist_top.append(np.sqrt((x_new_right[ix] - x_new_left[ix])**2 + (y_new_right[ix] - y_new_left[ix])**2))

        # for ix in randomlist_bottom:
        #     average_dist_bottom.append(np.sqrt((x_new_right[ix] - x_new_left[ix])**2 + (y_new_right[ix] - y_new_left[ix])**2))
        
        # average_dist_variance = np.var(average_dist_top + average_dist_bottom)

        # globalDiff = abs(abs(np.mean(average_dist_top) - np.mean(average_dist_bottom)) - previous.state["avg_previous_variation"])

        # topDiff = abs(np.mean(previous.state["previous_dist_variation_top"]) - np.mean(average_dist_top))

        # bottomDiff = abs(np.mean(previous.state["previous_dist_variation_bottom"]) - np.mean(average_dist_bottom))
        
        
        
        # if (average_dist_variance < 15000 or globalDiff < 50 or topDiff <50 or bottomDiff < 50):
        #     previous.add_prev_dist_variation(np.mean(average_dist_top),np.mean(average_dist_bottom))
        #     previous.update_prev_dist_variation()
        
        if(right_variance > 8000 ) and len(previous.state["previous_model_right"])!=0:
            x_new_right = previous.state["previous_model_right"]
            coef2 = previous.state["previous_coef_right"]
            if(right_variance > 8000):
                right_variance_high = True
        elif(right_variance < 8000):
            previous.update("previous_model_right",x_new_right)
            previous.update("previous_coef_right",f2)
            coef2 = f2
            

        if((left_variance > 8000 ) and len(previous.state["previous_model_left"])!=0):
            x_new_left = previous.state["previous_model_left"]
            coef1 = previous.state["previous_coef_right"]
            if(left_variance > 8000):
                left_variance_high = True
        elif(left_variance < 8000):
            previous.update("previous_model_left",x_new_left)
            previous.update("previous_coef_left",f1)
            coef1 = f1


        if(left_variance_high and right_variance_high):
            color = (0,0,255)
        elif(left_variance_high or right_variance_high):
            color = (0,255,255)
        else:
            color = (0,255,0)


        #color = (0,255,0)
        
        left = [list(pair) for pair in zip(x_new_left, y_new_left)]
        right = [list(pair) for pair in zip(x_new_right, y_new_right)]
        pts = np.vstack((left, right[::-1]))
        cv2.fillPoly(polyfill, np.int32([pts]), color)
        cv2.polylines(polyfill, np.int32([left]), isClosed=False, color=(255, 0, 255), thickness=10)
        cv2.polylines(polyfill, np.int32([right]), isClosed=False, color=(0, 0, 255), thickness=10)

        # # if(left_variance_high or right_variance_high):
        # #     print("Previous Model",np.var(previous.state["previous_model_left"]),np.var(previous.state["previous_model_right"]))
        # #     print("Distance variance",average_dist_variance)
        # #     print(left_variance,right_variance)
        #     #debug.display(plot=True)

        
        
        #debug.collect(polyfill,("image","gray"))

        
        warped = image.warpPerspective(polyfill,image.H_inv)
        overlaid_result = cv2.addWeighted(undistorted_img, 0.6, warped, 0.4, 0)

        left_roc = round(Line_fit.radius_of_curvature(coef1,x_new_left))
        right_roc = round(Line_fit.radius_of_curvature(coef2,x_new_right))

        if np.sign(left_roc) == np.sign(right_roc):
            mean = left_roc+right_roc/2
            if mean < 3900 and mean > 0:
                turn = "Turning Right"
            if np.abs(mean) > 3900:
                turn = "Moving Straight"
            if mean < 0 and mean > -3900:
                turn = "Turning Left"
        else: 
            turn = ""

        text = f'Radius of curvature: Left={left_roc} m, Right={right_roc} m'
        cv2.putText(overlaid_result, text, (40,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        text = turn
        cv2.putText(overlaid_result, text, (40,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        if(left_variance_high and right_variance_high):
            cv2.putText(overlaid_result, "Correcting both lanes", (40,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        elif(left_variance_high and not right_variance_high):
            cv2.putText(overlaid_result, "Correcting left lane", (40,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        elif(not left_variance_high and right_variance_high):
            cv2.putText(overlaid_result, "Correcting right lane", (40,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        else:
            cv2.putText(overlaid_result, "Both lanes are good", (40,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)

        
        debug.collect(overlaid_result,("image","gray"))
        #if(right_variance > 8000):
        #print(average_dist_variance)
        #debug.display(plot=True)
        
        #cv2.imshow("yellow mask",overlaid_result)
        out.write(overlaid_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)