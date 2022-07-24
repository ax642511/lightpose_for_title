import cv2
# 读取一张图片
img = cv2.imread("000000005134.jpg")
# 定义一个列表
point_list = [( 90, 250),
 (45 ,265),
 (30 ,250),
 (85, 270),
(140, 280),
(80 ,390),
(60, 465),
(235 ,560),
(95 ,390),
(80 ,510),
(75 ,240)
]

# 对读取的图片进行画圆，并保存到last_img

for i in point_list:
    last_img =cv2.circle(img,center = i,radius = 5,color = (255,0,3),thickness = 6)
# 显示图片
cv2.imshow ('for  circle',last_img)
# 保存图片
cv2.imwrite('./mediaed/1.jpg',last_img)
cv2.waitKey(0)
cv2.destroyAllWindows()