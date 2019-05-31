import random

def caculateBMI(heigh, weigh):
    bmi = weigh/((heigh/100)**2)
    if bmi < 18.5:
        return 'thin'
    elif bmi<25:
        return 'normal'
    else:
        return 'fat'

with open('bmi.csv','w',encoding='UTF-8') as file1:
    file1.write('heigh,weigh,label\n')
    category = {'thin':0,'normal':0,'fat':0}
    for i in range(3000):
        currentHeigh = random.randint(110,220)
        currentWeigh = random.randint(40,80)
        label = caculateBMI(currentHeigh,currentWeigh)
        category[label]=1
        file1.write("%d,%d,%s\n"%(currentHeigh,currentWeigh,label))
print("OK",category)