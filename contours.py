import cv2
import os


file_list=[]

for path,dirs,files in os.walk('image/imgs/1'):
    for name in files:
        file_list.append(int(name.split('.')[0]))
        print(name)

file_list.sort()

print("All names are saved in file_list")

for sinfile in file_list:
    src=os.path.join('image/imgs/1',str(sinfile)+'.png')        
    img=cv2.imread(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    print('bin get')
    # 保存二进制图片、寻找轮廓
    # cv2.imwrite('bin.png',binary)
    contours, hierarchy = cv2.findContours(255-binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 过滤轮廓长度
    con_res=[]
    for c in contours:
        if cv2.arcLength(c, True)<130:
            continue
        # if cv2.cv2.arcLength(c, True)>500:
        #     continue
        con_res.append(c)

    img_copy=img.copy()
    img_cir=img.copy()
    # 绘制全长轮廓
    # img = cv2.drawContours(img, con_res, -1, (255,0,0), 1)
    # cv2.imwrite('res.png',img)

    # 提取轮廓中的点
    some=[]
    for num in range(len(con_res)):
        some.append([])
        for i in range(0,len(con_res[num]),6):
            some[num].append(con_res[num][i])

    for index in range(len(some)):
        # img_some = cv2.drawContours(img_copy, some[index], -1, (255,0,0), 1)
        # cv2.imwrite('some.png',img_some)

        res=[]
        # 点
        for i in range(len(some[index])):
            res.append(some[index][i][0])
            img_cir=cv2.circle(img_cir,some[index][i][0],1,(0,0,255),-1)
        
        print('circle get')
    # img_cir=cv2.circle(img_cir,(1,10),1,(0,0,255),4)
    # cv2.imwrite('cir.png',img_cir)

        # 连点成线
        for i in range(len(res)-1):
            cv2.line(img_cir, res[i], res[i+1], (255,0,5), 1)
        cv2.line(img_cir, res[-1], res[0], (255,0,5), 1)
        
        print('lines get')
    
    with open('cir.txt','a') as f:
        f.write(str([x.tolist() for x in res]))   
        f.write('\r\n')
        
    print('save in txt') 
    cv2.imwrite(f'lines/{sinfile}_cir.png',img_cir)
    
    print('save pic success')
