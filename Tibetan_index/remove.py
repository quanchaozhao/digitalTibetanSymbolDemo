import os
import shutil
import matplotlib.pyplot as plt
origianPath = r'C:\Users\Zqc\Desktop\mark\第二次数据-寒假\AddBaseLine'  # 原始图像目录
temPath = r'C:\Users\Zqc\Desktop\mark\第二次数据-寒假\AddBaseLine1'  # 临时输出
# for i in os.listdir(origianPath):
#     if i.endswith('.xml'):
#         filename = os.path.join(origianPath, i.replace('xml', 'png'))
#         try:
#             file = open(filename)
#             print(i)
#         except FileNotFoundError:
#             plt.subplot(111)
#             plt.show
path = r'C:\Users\Zqc\Desktop\character\character'
path2 = r'C:\Users\Zqc\Desktop\character\data'
with open('Tibetan_symbol.txt') as readfile:
    filename = readfile.readline()[:-1]
    while(filename != None):
        try:
            shutil.move(os.path.join(path,filename + '.bmp'), os.path.join(path2,filename + '.bmp'))
        except FileNotFoundError:
            print(filename)
        finally:
            filename = readfile.readline()[:-1]