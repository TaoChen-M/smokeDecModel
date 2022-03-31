import os
ssr="image/imgs/1"
tosort=[]
res=[]
lines=open('change/image_index.txt','r').readlines()
tar='good.txt'
if os.path.exists(tar):
    os.remove(tar)
# print(lines)
for path,dir_list,file_list in os.walk(ssr):
    for name in file_list:
        tosort.append(int(name.split('.')[0]))
       
tosort.sort()

index=[lines[i-1] for i in tosort]


for i in range(len(tosort)):
    temp=str(tosort[i])+"   "+index[i]
    res.append(temp)
with open(tar,'w') as f:
    f.writelines(res)
