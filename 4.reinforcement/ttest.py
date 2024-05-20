import matplotlib.pyplot as mpl
import numpy as np
import scipy.stats as sp

def twosampleTTest(data1,data2):
    tStatistic, pValue = sp.ttest_ind(data1,data2)
    print("T statistic:",tStatistic)
    print("P value:",pValue)

    if pValue < 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")

def convergencePlot(ACInput,RInput,QInput):
    
    mpl.plot(range(1,len(ACInput)+1),ACInput,'r-',label='Actor Critic')
    mpl.plot(range(1,len(RInput)+1),RInput,'g-',label='REINFORCE')
    mpl.plot(range(1,len(QInput)+1),QInput,'b-',label='Approximate Q Learning')

    mpl.title('Comparing Convergence Behavior for Medium Classic')
    mpl.xlabel('Episode')
    mpl.ylabel('Total Reward on Episode \n(averaged over 100 runs)')
    mpl.legend()
    mpl.show()

def normalityAssumption(data1,data2):
    """Check normality of two samples using Shapiro-Wilk test."""

    _, p_value1 = sp.shapiro(data1)
    _, p_value2 = sp.shapiro(data2)
    print(p_value1)
    print(p_value2)
    if p_value1 > 0.05 and p_value2 > 0.05:
        print("Both samples are normally distributed")
    else:
        print("At least one sample is not normally distributed")

def read_text_file(filename):
    data = []
    with open(filename, 'r') as file:
        current_line = []
        for line in file:
            line = line.strip()
            if line:  
                current_line = [float(num) for num in line.split(',')]
            else:
                if current_line:
                    data.append(current_line)
                    current_line = []
        if current_line:
            data.append(current_line)
    return data

def averages(listOfLists):
    listArrays = [np.array(l)for l in listOfLists]
    stack = np.stack(listArrays)
    averages = np.mean(stack,axis=0)
    return averages

ACData = read_text_file('ACData.txt')
RData = read_text_file('RData.txt')
QData = read_text_file('QData.txt')
ACData = averages(ACData)
RData = averages(RData)
QData = averages(QData)

ACData2 = averages(read_text_file("AC2Data.txt"))
RData2 = averages(read_text_file("R2Data.txt"))
QData2 = averages(read_text_file("Q2Data.txt"))

convergencePlot(ACData,RData,QData)
convergencePlot(ACData2,RData2,QData2)


ACData_values = [598.4, 550.8, -485.1, 675.7, 678.1, 663.5, -501, 807.4, 850, 740.9]
RData_values = [-335.6, -453.3, -499.7, -443.1, 659.3, 772.2, -501, -384.6, -433.4, -446.4]
QData_values = [599.8, 669.9, 462.3, 777.6, 802.2, 775.3, -501, 686.9, 864, 836.1]


normalityAssumption(ACData_values,RData_values)
normalityAssumption(ACData_values,QData_values)
normalityAssumption(RData_values,QData_values)

twosampleTTest(ACData_values,RData_values)
twosampleTTest(ACData_values,QData_values)
twosampleTTest(RData_values,QData_values)