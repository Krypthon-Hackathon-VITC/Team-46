import matplotlib.pyplot as plt


with open('DataAnalytics.txt', 'r') as file:
    data = file.read()


count_0 = data.count('0')
count_1 = data.count('1')


total_count = count_0 + count_1
percentage_0 = (count_0 / total_count) * 100
percentage_1 = (count_1 / total_count) * 100


print("Number of 'unsatisfied customers':", count_0)
print("Number of 'satisfied customers':", count_1)
print("Percentage of 'unsatisfied customers': {:.2f}%".format(percentage_0))
print("Percentage of 'satisfied customers': {:.2f}%".format(percentage_1))


labels = ['unsatisfied customers', 'satisfied customers']
sizes = [percentage_0, percentage_1]
colors = ['#ff9999','#66b3ff']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Percentage of Unsatisfied and satisfied customers during data analysis')
plt.axis('equal')
plt.show()
