import cv2
import numpy as np
import exifread
# import tifffile as tiff
import pandas as pd
import torch
import numpy
from matplotlib import pyplot as plt


def getResolution(binary_path):
    f = open(binary_path, 'rb')
    tags = exifread.process_file(f)
    x = 0
    y = 0
    z = 0
    for tag in tags.keys():
        if (x != 0) and (y != 0) and (z != 0):
            break
        if "XResolution" in tag:
            x = (tags[tag].values[0].num * 1.0) / tags[tag].values[0].den
        if "YResolution" in tag:
            y = (tags[tag].values[0].num * 1.0) / tags[tag].values[0].den
        if "ImageDescription" in tag:
            array = (tags[tag]).values.split("\n")
            for i in range(0, array.__len__()):
                if "spacing" in array[i]:
                    z = float(array[i].split("=")[1])
    # Should return pixels per mm, which is > 1. We want mm per pixel, which should be < 1
    x = 1 / x
    y = 1 / y
    return x, y, z


def calculatePixels(botThreshold, topThreshold, maskedImage):
    thresh = maskedImage.copy()
    thresh[thresh > topThreshold] = 0
    thresh[thresh < botThreshold] = 0
    return np.count_nonzero(thresh)


def intensity(bot, top, img):  # calculate intensity
    thresh = img.copy()
    thresh[thresh > top] = 0
    thresh[thresh < bot] = 0
    # inten = 0
    # for i in range(thresh.__len__()):
    # if thresh[i] != 0:
    # inten += thresh[i]
    return np.sum(thresh)


def separatePixels(botThreshold, topThreshold, maskedImage):
    thresh = cv2.inRange(maskedImage, botThreshold, topThreshold)
    thresh[thresh > 0] = 255
    return thresh


def createMaskedImage(fullImage, maskImage):
    for i in range(0, fullImage.__len__()):
        temp1 = fullImage[i]
        temp2 = maskImage[i]
        temp1[temp2 <= 0] = 0
        fullImage[i] = temp1
    result = fullImage

    return result


def drawContour(contour, image):
    w, h = image.shape
    tempImage = image.copy()
    contour_mask = np.zeros((w, h), dtype=np.uint8)
    contour_mask = cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), -1)
    tempImage[contour_mask == 0] = 0

    return tempImage


def pixelCount(fullImage, maskImage, xRes, yRes, zRes, min_value, max_value, cList):
    # Run scan through calibration function
    fullImage = np.int16(fullImage)
    fullImage = fullImage - 1024
    # Create masked full image
    masked_im = createMaskedImage(fullImage, maskImage)

    vesselPixels = 0
    calcPixels = 0
    vesselIntense = 0
    calcIntense = 0

    maxRatio = 0
    maxRatioSlice = 0

    lesion_total = 0

    vlist = []
    clist = []
    vlist_intense = []
    clist_intense = []
    vsep = np.zeros((len(masked_im), 10))
    csep = np.zeros((len(masked_im), 10))

    for i in range(0, masked_im.__len__()):  # For each slice
        vvcount = 0
        cccount = 0
        vvintense = 0
        ccintense = 0
        t = 0
        for j in cList[i]:  # For each contour within that slice
            contour_section = drawContour(j, masked_im[i])
            # Calculate number of vessel pixels
            newVPixels = calculatePixels(min_value, max_value, contour_section)
            vesselPixels += newVPixels
            vvcount += newVPixels
            vsepcount = newVPixels

            newintensev = intensity(min_value, max_value, contour_section)
            vesselIntense += newintensev
            vvintense += newintensev

            # Calculate number of calcification pixels
            newCPixels = calculatePixels(max_value + 1, 3000, contour_section)
            calcPixels += newCPixels
            cccount += newCPixels
            csepcount = newCPixels

            newintensec = intensity(max_value + 1, 3000, contour_section)
            calcIntense += newintensec
            ccintense += newintensec

            # Calculate ratio and max blockage slice
            if newCPixels != 0 and newVPixels != 0:
                newRatio = newCPixels / newVPixels
                if newRatio > maxRatio:
                    maxRatio = newRatio
                    maxRatioSlice = i + 1

            vsep[i][t] = vsepcount
            csep[i][t] = csepcount
            t += 1

        vlist.append(vvcount)
        clist.append(cccount)
        vlist_intense.append(vvintense)
        clist_intense.append(ccintense)
        # Calculate number of lesions
        lesions = separatePixels(max_value + 1, 3000, masked_im[i])
        contours, hier = cv2.findContours(lesions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_new = []
        for j in range(contours.__len__()):
            if cv2.contourArea(contours[j]) > 1:
                contours_new.append(contours[j])
        lesion_total += contours_new.__len__()

    # Print number of pixels
    print("\nNumber of vessel pixels: " + str(vesselPixels))
    print("Number of calcification pixels: " + str(calcPixels))
    print("\nSlice with maximum blockage: " + str(maxRatioSlice))

    # Print volume information
    vesselVolume = vesselPixels * zRes * yRes * xRes
    calcVolume = calcPixels * zRes * yRes * xRes
    print ("vessV: ", vesselPixels)
    print ("calcP: ", calcPixels)
    print ("calcV: ", calcVolume)
    percentage = calcVolume / (vesselVolume + calcVolume) * 100

    print("\nVessel volume: " + str(vesselVolume) + " mm^3")
    print("Calc volume: " + str(calcVolume) + " mm^3")
    print("\nPercentage of vessels that contain calcification: " + str(percentage) + "%")

    print("\nNumber of lesions: " + str(lesion_total))

    #
    vpixels = pd.DataFrame(data=vlist)
    #
    vpixels.to_csv('vpixels.csv', encoding='gbk')

    #
    cpixels = pd.DataFrame(data=clist)
    #
    cpixels.to_csv('cpixels.csv', encoding='gbk')

    #
    vinten = pd.DataFrame(data=vlist_intense)
    #
    vinten.to_csv('vintense.csv', encoding='gbk')

    #
    cinten = pd.DataFrame(data=clist_intense)
    #
    cinten.to_csv('cintense.csv', encoding='gbk')

    #
    data1 = pd.DataFrame(csep)
    #
    data1.to_csv('csep.csv')

    #
    data2 = pd.DataFrame(vsep)
    #
    data2.to_csv('vsep.csv')


def superimpose_color(rgb_image, original_image, mask, min_value, max_value):
    # Create red and blue images
    u = rgb_image[0].shape  #############
    h = u[1]
    w = u[0]

    # src = rgb_image[0]        #######
    # rgb_image[0] = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)   #########
    # for j in range(1, 446):
    # src2 = rgb_image[j]        #######
    # rgb_image[j] = cv2.cvtColor(src2, cv2.COLOR_GRAY2BGR)

    redImage = np.zeros((w, h, 3), np.uint8)
    redImage[:] = (255, 0, 0)  #######B,G,R

    blueImage = np.zeros((w, h, 3), np.uint8)
    blueImage[:] = (0, 0, 255)

    gImage = np.zeros((w, h, 3), np.uint8)  #####
    gImage[:] = (0, 255, 0)  #########

    # Create masked full image
    original_image = original_image - 1024
    masked_im = createMaskedImage(original_image, mask)

    vesselThresh = masked_im.copy()
    calcThresh = masked_im.copy()
    wallThresh = masked_im.copy()
    totalThresh = masked_im.copy()
    totalThresh_inv = masked_im.copy()
    # rgb = rgb_image.copy()
    for i in range(0, masked_im.__len__()):
        # Separate vessel pixels
        vesselThresh[i] = separatePixels(min_value, max_value, masked_im[i])
        vesselThresh1 = vesselThresh[i].astype(np.uint8)

        # Separate calcification pixels
        calcThresh[i] = separatePixels(max_value + 1, 3000, masked_im[i])
        calcThresh1 = calcThresh[i].astype(np.uint8)

        # Separate vessel wall pixels
        wallThresh[i] = separatePixels(100, 385, masked_im[i])  ########
        wallThresh1 = wallThresh[i].astype(np.uint8)  ############

        # Create inverse masks
        # totalThresh[i] = cv2.bitwise_or(vesselThresh[i], calcThresh[i])
        totalThresh[i] = cv2.bitwise_or(vesselThresh[i], calcThresh[i], wallThresh[i])  ###########
        totalThresh_inv[i] = cv2.bitwise_not(totalThresh[i])
        totalThresh_inv1 = totalThresh_inv[i].astype(np.uint8)

        # Black out regions of masks in frame
        rgb_image[i] = cv2.bitwise_and(rgb_image[i], rgb_image[i], mask=totalThresh_inv1)
        # cv2.imshow('rgb',rgb_image[i])
        # cv2.waitKey(0)
        # print('rgb_image[i].shape',rgb_image[i].shape)  #(512,512,3)
        # Get regions of interest of colored arrays
        redImage_mask = cv2.bitwise_and(redImage, redImage, mask=calcThresh1)
        # cv2.imshow('red',redImage_mask)
        # cv2.waitKey(0)
        blueImage_mask = cv2.bitwise_and(blueImage, blueImage, mask=vesselThresh1)
        # print(vesselThresh1.shape)
        # print(blueImage.shape)
        # Add results together
        rgb_image[i] = rgb_image[i] + redImage_mask  # +1024  test2  #cv2.add(rgb_image[i], redImage_mask)
        # print('redImage_mask', redImage_mask.shape)  #(512,512)
        rgb_image[i] = rgb_image[i] + blueImage_mask  # +1024 #cv2.add(rgb_image[i], blueImage_mask)

        # r,g,b,a, = cv2.split(rgb_image[i])
        # new = cv2.merge("RGB",(r,g,b))
        gImage_mask = cv2.bitwise_and(gImage, gImage, mask=wallThresh1)
        rgb_image[i] = rgb_image[i] + gImage_mask
        # rgb[i] = cv2.cvtColor(rgb_image[i],cv2.COLOR_GRAY2BGR)
    # CTA3 = []
    # ret3, im3 = cv2.imreadmulti(filename=rgb_image, mats=CTA3)
    # for j in range(0, 446):
    # src2 = rgb_image[j]        #######
    # cv2.cvtColor(src2,rgb_image[j], cv2.COLOR_BGR2GREAY)
    # rgb_image[j] = cv2.cvtColor(src2, cv2.COLOR_BGR2HSI)
    # print(rgb_image.shape)
    # np.array()

    # rgb_image = rgb_image+1024
    return rgb_image
