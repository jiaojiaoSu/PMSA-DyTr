import xlwt#写入文件
import xlrd#打开excel文件
import os
fileName='log_all\detailed_log.txt'
fo=open(fileName,encoding='UTF-8')
lines=fo.readlines()
lis=[]

#新建一个excel文件
file=xlwt.Workbook(encoding='utf-8',style_compression=0)
#新建一个sheet
sheet=file.add_sheet('data')

counter=0
for line in lines:
    if  ' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] ' in line:
        # print(line[-6:-1])
        val=float(line[-6:-1])
        lis.append(val)
        sheet.write(counter,0,val)
        counter=counter+1

print(lis)
file.save('record1.xls')
fo.close()
# if not os.path.exists( excelName):  # 判断是否存在文件夹如果不存在则创建为文件夹
#         os.makedirs( excelName)
