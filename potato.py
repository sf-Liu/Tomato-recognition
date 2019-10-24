# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import csv

def gradian(input): #计算梯度
    grad_x =cv.Sobel(input, cv.CV_16S, 1, 0)
    grad_y =cv.Sobel(input, cv.CV_16S, 0, 1)
    gradImage = abs(grad_x) + abs(grad_y)

    cv.namedWindow("梯度算子边缘图", 0)
    cv.imshow("梯度算子边缘图", gradImage)
    return gradImage

def mythreshold(input):
    ret,thresholdedImage=cv.threshold( input,  20, 255, cv.THRESH_OTSU)
    thresholdedImage=255-thresholdedImage
    cv.namedWindow("threshold", 0)
    cv.imshow("threshold", thresholdedImage)
    #cv.imwrite("阈值化后的二值图hui.tif", thresholdedImage)
    return thresholdedImage

def findruler(input):
    [width, height] = input.shape
    length=min(width,height)
    lines = cv.HoughLinesP(input, 1, np.pi/720 , 30, minLineLength=length*0.3, maxLineGap=20)
    lines1 = lines[:, 0, :]  # 提取为二维
    image=input*0
    for x1, y1, x2, y2 in lines1[:]:
        cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv.namedWindow("findruler1", 0)
    cv.imshow("findruler1", image)
    m=0
    i=0
    for x1, y1, x2, y2 in lines1[:]:
        k=abs((y1-y2)/(x1-x2))
        if k>10:
            lines1[m]=lines1[i]
            m+=1
        i+=1
    if m==0:
        print('the board is slant')
        return -1

    def by_x1(t):
        return (t[0])
    lines2 = sorted(lines1[0:m], key=by_x1)
    i=0
    for j in range(0,m-1):
        deltax1 =abs( lines2[j][0] - lines2[j+1][0])
        deltax2 =abs( lines2[j][2] - lines1[j+1][2])
        if deltax1>20 and  deltax2>20 :
           lines2[i]=lines2[j]
           i+=1
    lines3= lines2[0:i]
    distance=np.empty(i-1,dtype=int)
    for j in range(0,i-1):
        distance[j]= abs( lines3[j][0] - lines3[j+1][0])
    counts = np.bincount(distance)
    measure=np.argmax(counts)
    result=input * 0
    for x1, y1, x2, y2 in lines3[:]:
        cv.line(result, (x1, y1), (x2, y2), (255, 0, 0), 1)
    print(i)
    cv.namedWindow("ruler", 0)
    cv.imshow("ruler", result)
    return measure

def denoise(input):
    de=cv.GaussianBlur( input, (15, 15), 0)
    cv.namedWindow("denoise", 0)
    cv.imshow("denoise", de)
    return de

def greenMask(input):
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    # change to hsv model
    hsv = cv.cvtColor(input,cv.COLOR_BGR2HSV)
    # get mask
    mask = cv.inRange(hsv, lower_green, upper_green)
    mask=255-mask
    cv.namedWindow("Mask", 0)
    cv.imshow('Mask', mask)

    #res = cv.bitwise_and(input, input, mask=mask)
    #cv.namedWindow("Mask1", 0)
    #cv.imshow('Mask1', res)
    return mask

def yellowMask(input):
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    # change to hsv model
    hsv = cv.cvtColor(input, cv.COLOR_BGR2HSV)
    # get mask
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    cv.namedWindow("Mask", 0)
    cv.imshow('Mask', mask)
    #cv.imwrite('black.jpg', mask)

    res = cv.bitwise_and(input, input, mask=mask)
    cv.namedWindow("Mask1", 0)
    cv.imshow('Mask1', res)
    return mask

def shape(input,type):
    element = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    shapeIm = cv.morphologyEx( input, type, element)
    cv.namedWindow("shape", 0)
    cv.imshow("shape", shapeIm)
    return shapeIm

def cut(input,scale):
    if scale>1 or scale<=0:
        print('scale error')
        return -1
    [width,height]=input.shape
    left=round(width*(1-scale)/2)
    right=round(width*(1+scale)/2)
    up=round(height*(1+scale)/2)
    down=round(height*(1-scale)/2)
    result=input[left:right,down:up]
    cv.namedWindow("cut", 0)
    cv.imshow("cut", result)
    return result

def cut_color(input,scale):
    if scale>1 or scale<=0:
        print('scale error')
        return -1
    [width,height,depth]=input.shape
    left=round(width*(1-scale)/2)
    right=round(width*(1+scale)/2)
    up=round(height*(1+scale)/2)
    down=round(height*(1-scale)/2)
    result=input[left:right,down:up,:]
    cv.namedWindow("cut_color", 0)
    cv.imshow("cut_color", result)
    return result

def measure(input,input1,ruler):
    [width,height]=input.shape
    stepx=round(width/11)
    stepy=round(height/11)
    line=np.empty((40,4),dtype=int)
    delta=np.empty(40)
    p=0
    for j in range(1,10):
        temp = input[:,stepy*j]
        flag=0
        for i in range(0,width):
            x=temp[i]
            if x==255 and flag==0 :
                begin=i
                flag=1
                k=0
            elif x==255 and flag==1:
                if k<15:
                    k+=1
                else:
                    flag=-1
            elif x==0 and  flag==1:
                flag = 2
                k=0
            elif x==0 and flag==2:
                if k<50:
                    k+=1
                else:
                    flag = -2
            elif x==255 and flag==2:
                flag=3
                k=0
            elif x==255 and flag==3:
                if k<15:
                    k+=1
                else:
                    flag = -1
            elif x==0 and flag==3:
                end=i-k
                flag=0
                line[p]=(stepy*j,begin,stepy*j,end)
                delta[p]=end-begin
                p+=1
            elif x == 0 and flag == -1:
                flag = 0
            elif x == 255 and flag == -2:
                flag = 0
    for j in range(1,10):
        temp = input[stepx * j,:]
        flag=0
        for i in range(0,height):
            x=temp[i]
            if x==255 and flag==0 :
                begin=i
                flag=1
                k=0
            elif x==255 and flag==1:
                if k<15:
                    k+=1
                else:
                    flag=-1
            elif x==0 and  flag==1:
                flag = 2
                k=0
            elif x==0 and flag==2:
                if k<50:
                    k+=1
                else:
                    flag = -2
            elif x==255 and flag==2:
                flag=3
                k=0
            elif x==255 and flag==3:
                if k<15:
                    k+=1
                else:
                    flag = -1
            elif x==0 and flag==3:
                end=i-k
                flag=0
                line[p] = (begin, stepx * j, end, stepx * j)
                delta[p] = end - begin
                p+=1
            elif x==0 and flag==-1:
                flag=0
            elif x==255 and flag==-2:
                flag=0
    line1=line[0:p-1]
    delta1=delta[0:p-1]
    i=0
    temp=input1
    for x1, y1, x2, y2 in line1[:]:
        cv.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 1)
        result = cv.putText(temp, str(i), (x2+10, y2+10), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,0, 255),2)
        i+=1

    cv.namedWindow("measure", 0)
    cv.imshow("measure", result)
    cv.imwrite("stem-measure.jpg", result)

    tab=np.zeros((p-1,3))
    for i in range(0,p-1):
        tab[i,0]=i
        tab[i,1]=delta1[i]
        tab[i,2]=tab[i,1]/ruler
    with open('stem-measure.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(tab)


def skeleton4(input):
    distance=input/255
    [width,height]=distance.shape
    for i in range(1,height):
        for j in range(1,width):
            if distance[j,i]>0:
                left=distance[j-1,i]
                up=distance[j,i-1]
                distance[j,i]=min(left,up)+1
    for i in range(height - 2,1,-1):
        for j in range(width - 2, 1,-1):
            if distance[j,i]>0:
                right=distance[j+1,i]
                down=distance[j,i+1]
                temp=min(right,down)+1
                distance[j, i]=min(temp,distance[j,i])
    skeleton = input * 0
    for i in range(1,height - 1):
        for j in range( 1,width - 1):
            if distance[j, i] > 0:
                a = distance[j, i]
                left = distance[j - 1, i]
                up = distance[j, i - 1]
                right = distance[j + 1, i]
                down = distance[j, i + 1]
                if a >= left and a >= right and a >= up and a >= down and a<50:
                    skeleton[j, i] = 255
    distance=distance/50
    cv.namedWindow("distance", 0)
    cv.imshow("distance", distance)
    cv.namedWindow("skeleton", 0)
    cv.imshow("skeleton", skeleton)
    return skeleton

def straited(input,org):
    [width, height] = input.shape
    length = min(width, height)
    lines0 = cv.HoughLinesP(input, 1, np.pi / 720, 30, minLineLength=length * 0.05, maxLineGap=LINE_CAPTURE)
    lines1 = lines0[:, 0, :]  # 提取为二维
    image = input * 0
    i = 0
    for x1, y1, x2, y2 in lines1[:]:
        cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv.putText(image, str(i), (x2 + 10, y2 + 10), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        i=i+1
    cv.namedWindow("strait1", 0)
    cv.imshow("strait1", image)

    num,i = lines1.shape
    i = 0
    lines2=np.empty((num,5))
    for x1, y1, x2, y2 in lines1[:]:
        theta = np.arctan((y1 - y2) / (x1 - x2))
        lines2[i][0:4]=lines1[i][:]
        lines2[i][4]=theta
        i=i+1
    def by_x5(t):
        return (t[4])
    lines2 = sorted(lines2[:], key=by_x5)
    j=0
    for i in range(0,num-1):
        if lines2[i+1][4]-lines2[i][4]<0.17:  #0.17大概是10度
            b1=lines2[i][2]-(lines2[i][0]-lines2[i][2])/(lines2[i][1]-lines2[i][3])*lines2[i][3]
            b2=lines2[i+1][2]-(lines2[i+1][0]-lines2[i+1][2])/(lines2[i+1][1]-lines2[i+1][3])*lines2[i+1][3]
            dis=abs(b2-b1)*np.cos(lines2[i][4])
            if dis<10:
                lines2[i][0]=-1
                j=j+1
    lines3 =  np.empty((num-j,5))
    j=0
    for i in range(0,num):
        if lines2[i][0]!=-1:
            lines3[j]=lines2[i]
            j=j+1
    lines1=lines3[:,0:4].astype(int)
    i=0
    for x1, y1, x2, y2 in lines1[:]:
        cv.line(org, (x1, y1), (x2, y2), (255, 0, 0), 1)
        image=cv.putText(org, str(i), (x2 + 10, y2 + 10), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        i=i+1
    cv.namedWindow("angle-measure", 0)
    cv.imshow("angle-measure", image)
    cv.imwrite("angle-measure.jpg", image)

    tab=np.zeros((i,i),dtype=int)
    for j in range(0,i-1):
        tab[0,j]=tab[j+1,0]=j
    tab[0, 0] = 999
    tab[0, i-1] = i
    for j in range(0,i):
        for k in range(j+1,i):
            temp=abs(lines3[j][4]-lines3[k][4])/np.pi*180
            if temp>90: temp=180-temp
            tab[j+1,k]=temp
    with open('angle-measure.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(tab)


LINE_CAPTURE=12      #可以通过调节该参数调整对植物枝杈识别的敏感度，推荐介于10~20

def main():
    image_g = cv.imread("1.jpg",cv.IMREAD_GRAYSCALE)
    image = cv.imread("1.jpg")
    cv.namedWindow("src", 0)
    cv.imshow("src", image)

    image_thresh = mythreshold(image_g)
    image_mask=greenMask(image)
    image_shape = shape(image_mask, 0)
    image_measure = cv.bitwise_and(image_thresh, image_thresh, mask=image_shape)
    image_cut = cut(image_measure,0.8)
    ruler = findruler(image_cut)

    image_noline = shape(image_thresh, 2)
    image_shape = shape(image_mask, 1)
    image_stem = cv.bitwise_and(image_noline, image_noline, mask=image_shape)
    cv.namedWindow('res',0)
    cv.imshow('res',image_stem)
    measure(cut(image_stem, 0.8),cut_color(image,0.8),ruler)

    image_sk=skeleton4(255-image_mask)
    image = cv.imread("1.jpg")
    straited(cut(image_sk,0.8),cut_color(image,0.8))
    '''''
    image = cv.imread("2.jpg")
    image_mask=yellowMask(image)
    image_mask=cv.bitwise_and(image, image, mask=image_mask)
    cv.imwrite("color_config.jpg", image_mask)
    '''
    cv.waitKey(0)
    return 0



if __name__ == '__main__':
    main()