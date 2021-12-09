import cv2
import numpy as np

def generate_Mask(img_path,mask_save_path,cut_save_path):
    # read image
    img = cv2.imread(img_path)
    hh, ww = img.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2

    # define circles
    radius1 = 25
    radius2 = 75
    xc = hh // 2
    yc = ww // 2
    
    # prepare white iamge
    white = np.zeros_like(img)
    white[:] = (255,255,255)

    # draw filled circles in white on black background as masks
    mask1 = np.zeros_like(img)
    mask1 = cv2.circle(mask1, (xc,yc), radius1, (255,255,255), -1)
    mask2 = np.zeros_like(img)
    mask2 = cv2.circle(mask2, (xc,yc), radius2, (255,255,255), -1)

    # subtract masks and make into single channel
    mask = cv2.subtract(mask2, mask1)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    invert_mask = cv2.bitwise_not(mask)
    # put mask on image
    result_mask_part = cv2.bitwise_and(white,white,mask=mask)
    result_image_part = cv2.bitwise_and(img,img,mask=invert_mask)
    result = cv2.add(result_image_part,result_mask_part)

    # save results
    # cv2.imwrite('lena_mask1.png', mask1)
    # cv2.imwrite('lena_mask2.png', mask2)
    cv2.imwrite(mask_save_path, mask)
    cv2.imwrite(cut_save_path,result)

    cv2.imshow('mask', white)
    cv2.imshow('masked image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mask_save_path = "../mask/sample.png"
    cut_save_path = "../image/masked_sample.png"
    img_path = "../image/sample.png"
    generate_Mask(img_path,mask_save_path,cut_save_path)